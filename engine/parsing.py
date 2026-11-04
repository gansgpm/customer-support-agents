"""Robust parsing + normalisation of the agents' JSON outputs."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

CATEGORIES = ["Shipping", "Refund", "Account", "Product", "Billing", "Service", "General"]
URGENCIES = ["Low", "Medium", "High"]
SENTIMENTS = ["Positive", "Neutral", "Negative", "Angry"]

_CATEGORY_SYNONYMS = {
    "delivery": "Shipping", "shipping": "Shipping", "logistics": "Shipping", "order status": "Shipping",
    "return": "Refund", "returns": "Refund", "refund": "Refund", "exchange": "Refund", "cancellation": "Refund",
    "account": "Account", "login": "Account", "security": "Account", "password": "Account",
    "product": "Product", "replacement": "Product", "warranty": "Product", "quality": "Product", "damaged": "Product",
    "billing": "Billing", "payment": "Billing", "payments": "Billing", "invoice": "Billing", "promo": "Billing",
    "service": "Service", "installation": "Service", "repair": "Service", "technician": "Service",
    "claim": "Service", "membership": "Service", "subscription": "Service", "protection": "Service",
    "general": "General", "other": "General", "sales": "General", "inquiry": "General", "enquiry": "General",
}


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of an LLM message (fenced or bare)."""
    if not text:
        return None
    candidates: List[str] = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for cand in candidates:
        for attempt in (cand, _repair(cand)):
            try:
                obj = json.loads(attempt, strict=False)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"(?<=[{,\s])'([^']+)'\s*:", r'"\1":', s)  # single-quoted keys
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    return s


def _pick(value: Any, allowed: List[str], default: str) -> str:
    v = str(value or "").strip()
    for a in allowed:
        if v.lower() == a.lower():
            return a
    for a in allowed:
        if a.lower() in v.lower():
            return a
    return default


def normalize_category(value: Any) -> str:
    v = str(value or "").strip().lower()
    for c in CATEGORIES:
        if v == c.lower():
            return c
    for key, cat in _CATEGORY_SYNONYMS.items():
        if key in v:
            return cat
    return "General"


def normalize_triage(obj: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj or {}
    sentiment = obj.get("sentiment") or signal.get("sentiment", "Neutral")
    if str(sentiment).lower() in {"very angry", "furious", "angry"}:
        sentiment = "Angry"
    score = obj.get("sentiment_score", signal.get("sentiment_score", 0.0))
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = float(signal.get("sentiment_score", 0.0))
    risk = obj.get("risk_flags", signal.get("risk_flags", []))
    if isinstance(risk, str):
        risk = [r.strip() for r in risk.split(",") if r.strip()]
    risk = sorted(set(risk) | set(signal.get("risk_flags", [])))
    try:
        priority = int(float(obj.get("priority_score", 0)))
    except (TypeError, ValueError):
        priority = 0
    return {
        "category": normalize_category(obj.get("category")),
        "urgency": _pick(obj.get("urgency"), URGENCIES, "Medium"),
        "sentiment": _pick(sentiment, SENTIMENTS, "Neutral"),
        "sentiment_score": round(max(-1.0, min(1.0, score)), 2),
        "summary": str(obj.get("summary", "")).strip()[:240],
        "standalone_query": str(obj.get("standalone_query", "") or "").strip()[:300],
        "intent": str(obj.get("intent", "")).strip()[:160],
        "language": str(obj.get("language") or signal.get("language", "en")),
        "risk_flags": risk,
        "repeat_issue": bool(obj.get("repeat_issue", False)),
        "similar_tickets": obj.get("similar_tickets", []) if isinstance(obj.get("similar_tickets"), list) else [],
        "priority_score": max(0, min(100, priority)),
        "urgency_reason": str(obj.get("urgency_reason", "")).strip()[:200],
        "product_line": str(obj.get("product_line", "") or "").strip(),
        "product_lines": _as_list(obj.get("product_lines", [])),
        "product_or_service": str(obj.get("product_or_service", "") or "").strip()[:120],
    }


def _as_list(value) -> List[str]:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def normalize_kb(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = obj or {}
    sources = obj.get("sources", [])
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    facts = obj.get("key_facts", [])
    if not isinstance(facts, list):
        facts = []
    clean_facts = []
    for f in facts:
        if isinstance(f, dict) and f.get("fact"):
            clean_facts.append({"ref": str(f.get("ref", "")).strip(), "source": str(f.get("source", "")),
                                "fact": str(f["fact"]).strip()})
        elif isinstance(f, str) and f.strip():
            clean_facts.append({"source": "", "fact": f.strip()})
    return {
        "match_found": bool(obj.get("match_found", bool(clean_facts))),
        "sources": [str(s) for s in sources],
        "key_facts": clean_facts,
        "similar_resolutions": obj.get("similar_resolutions", []) if isinstance(obj.get("similar_resolutions"), list) else [],
        "context_brief": str(obj.get("context_brief", "")).strip(),
        "gaps": str(obj.get("gaps", "")).strip(),
    }


def normalize_resolution(obj: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
    if not obj:  # model ignored the JSON format – keep the text, estimate nothing
        m = re.search(r"confidence\D{0,5}(\d{1,3})", raw_text or "", flags=re.I)
        return {"response": (raw_text or "").strip(), "confidence": int(m.group(1)) if m else 50,
                "used_sources": [], "citations": [], "answer_found": True, "parse_fallback": True}
    try:
        conf = float(obj.get("confidence", 50))
        conf = conf * 100 if 0 < conf <= 1 else conf
    except (TypeError, ValueError):
        conf = 50
    sources = obj.get("used_sources", [])
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    citations = obj.get("citations", [])
    if isinstance(citations, str):
        citations = re.findall(r"[SL]\d+", citations)
    return {
        "citations": [str(c).strip().strip("[]") for c in citations if str(c).strip()],
        "response": str(obj.get("response", "")).strip(),
        "confidence": int(max(0, min(100, round(conf)))),
        "used_sources": [str(s) for s in sources],
        "answer_found": bool(obj.get("answer_found", True)),
        "language": str(obj.get("language", "")),
    }
