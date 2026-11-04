"""Product-line & service catalog (knowledge_base/catalog.json).

The catalog is the single place that describes what the company sells and
which support desk owns it. Add a product line or service by adding an entry
and its policy document - no code changes are needed:

* ``keywords``        - how the Customer Query Agent's ``identify_product_line`` tool recognises it
* ``documents``       - the knowledge-base files that belong to it (scoped RAG)
* ``team``            - the specialist desk that receives its escalations (+ env vars for contacts)
* ``risk_keywords``   - line-specific risks that always escalate (battery swelling, food safety...)
* ``sla``             - optional response-time overrides used in the customer's holding message
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

GENERAL = "general"


@dataclass
class Line:
    id: str
    type: str  # general | product_line | service
    name: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    team: Dict[str, str] = field(default_factory=dict)
    risk_keywords: Dict[str, List[str]] = field(default_factory=dict)
    sla: Dict[str, str] = field(default_factory=dict)
    exclude: List[str] = field(default_factory=list)  # phrases that look like a keyword but are not ("phone number")

    @property
    def team_name(self) -> str:
        return self.team.get("name", "General Support Desk")


def _contact(team: Dict[str, str], key: str) -> str:
    env = team.get(f"{key}_env", "")
    return os.getenv(env, "").strip() if env else ""


class Catalog:
    def __init__(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.company = data.get("company", "")
        self.general = Line(**{k: v for k, v in data["general"].items() if k in Line.__dataclass_fields__})
        self.lines: List[Line] = [Line(**{k: v for k, v in d.items() if k in Line.__dataclass_fields__})
                                  for d in data.get("lines", [])]
        self.risk_teams: Dict[str, Dict[str, str]] = data.get("risk_teams", {})
        self._by_id = {l.id: l for l in [self.general] + self.lines}
        self._doc_scope = {doc: l.id for l in [self.general] + self.lines for doc in l.documents}
        self._patterns = {
            l.id: [(kw, re.compile(rf"\b{re.escape(kw.lower())}(?:s|es)?\b")) for kw in l.keywords] for l in self.lines
        }

    # ---------------------------------------------------------------- lookup
    def ids(self, include_general: bool = True) -> List[str]:
        return ([GENERAL] if include_general else []) + [l.id for l in self.lines]

    def get(self, line_id: str) -> Line:
        return self._by_id.get((line_id or "").strip().lower(), self.general)

    def name(self, line_id: str) -> str:
        return self.get(line_id).name

    def normalize(self, value) -> str:
        """Map an LLM answer ('Home Appliances', 'electronics', 'Remya Care') to a catalog id."""
        v = str(value or "").strip().lower()
        if v in self._by_id:
            return v
        for l in self.lines:
            if v and (v == l.name.lower() or v.replace(" ", "_") == l.id or v in l.name.lower() or l.id in v):
                return l.id
        return GENERAL

    def scope_of(self, document: str) -> str:
        return self._doc_scope.get(Path(document).name, GENERAL)

    # -------------------------------------------------------------- identify
    def identify(self, text: str, hint: str = "") -> Dict:
        """Keyword-based detection of every product line / service mentioned in a message."""
        lower = (text or "").lower()
        scores = []
        for line in self.lines:
            text_for_line = lower
            for phrase in line.exclude:
                text_for_line = text_for_line.replace(phrase.lower(), " ")
            hits = sorted({kw for kw, pat in self._patterns[line.id] if pat.search(text_for_line)})
            if hits:
                distinct = len({re.sub(r"s$", "", h) for h in hits})  # "shoe" + "shoes" count once
                weight = distinct + (1.0 if line.type == "service" else 0.0)  # a failed service is usually the issue
                scores.append((weight, line.type != "service", line.id, hits))
        scores.sort(key=lambda x: (-x[0], x[1], x[2]))
        scores = [(w, lid, hits) for w, _, lid, hits in scores]
        detected = [{"id": lid, "name": self.name(lid), "type": self.get(lid).type, "matched": hits}
                    for _, lid, hits in scores]
        hint_id = self.normalize(hint) if hint and hint.lower() not in {"auto", "auto-detect", ""} else ""
        order = [d["id"] for d in detected]
        if hint_id and hint_id != GENERAL:
            order = [hint_id] + [i for i in order if i != hint_id]
        primary = order[0] if order else GENERAL
        risks = []
        for lid in order:
            for flag, phrases in self.get(lid).risk_keywords.items():
                found = [p for p in phrases if p in lower]
                if found:
                    risks.append({"flag": flag, "line": lid, "matched": found})
        return {
            "primary": primary,
            "primary_name": self.name(primary),
            "all": order or [GENERAL],
            "detected": detected,
            "customer_selected": hint_id or "",
            "line_risks": risks,
            "team": self.get(primary).team_name,
        }

    # --------------------------------------------------------------- routing
    def route(self, primary: str, risk_flags: List[str]) -> Dict[str, str]:
        """Which desk owns the ticket: fraud / payment / legal desks first, else the product-line desk."""
        for flag in ("fraud", "prompt_injection", "payment_dispute", "legal_threat"):
            if flag in risk_flags and flag in self.risk_teams:
                t = self.risk_teams[flag]
                return {"name": t["name"], "email": _contact(t, "email"), "telegram": _contact(t, "telegram"),
                        "reason": flag}
        line = self.get(primary)
        return {"name": line.team_name, "email": _contact(line.team, "email"),
                "telegram": _contact(line.team, "telegram"), "reason": f"product line: {line.id}"}

    def sla(self, line_id: str, urgency: str) -> Optional[str]:
        return self.get(line_id).sla.get(urgency)

    def prompt_table(self) -> str:
        rows = [f"- general: {self.general.description}"]
        for l in self.lines:
            rows.append(f"- {l.id} ({'service' if l.type == 'service' else 'product line'}: {l.name}): {l.description}")
        return "\n".join(rows)

    def all_teams(self) -> List[str]:
        teams = [self.general.team_name] + [l.team_name for l in self.lines]
        teams += [t["name"] for t in self.risk_teams.values()]
        return list(dict.fromkeys(teams))


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> Catalog:
    return Catalog(Path(path))


def get_catalog(kb_dir: Path) -> Catalog:
    path = Path(kb_dir) / "catalog.json"
    return _load(str(path), path.stat().st_mtime)
