"""Post-generation grounding check for the customer reply (runs on every ticket, not just in the evaluation).

The Resolution Agent is *told* to use only retrieved facts; this module *checks* it:

* **Citations** - every passage the Knowledge Base Agent saw has a reference (S1, S2, ...). Citations that
  point to a passage that was never retrieved are removed; the remaining ones are shown with the reply.
* **Figures** - every number in the reply (amounts, days, hours, percentages) must appear in the retrieved
  passages (or in the customer's own message / the ticket and order ids). A reply that states a figure the
  documents do not contain is flagged as ungrounded, and the ticket is escalated so a human checks it.
* **Support** - share of the reply's factual sentences whose key words are found in the retrieved text
  (a lexical faithfulness score, also reported by the evaluation).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from rag.text import content_terms, split_sentences, tokenize

NUMBER = re.compile(r"(?<![\w-])(\d+(?:[.,]\d+)*)(?![\w-])")
# Sentences that are courtesy, not claims (greeting, empathy, ticket reference, sign-off)
NON_CLAIM = re.compile(r"^(hi|hello|dear|thank|thanks|sorry|i'?m sorry|we'?re sorry|i understand|"
                       r"your ticket number|ticket number|if anything is still unclear|best regards|regards|-\s)",
                       re.I)


def numbers(text: str) -> List[str]:
    return [n.replace(",", "") for n in NUMBER.findall(text or "")]


def _claim_sentences(reply: str, signatures: List[str] = ()) -> List[str]:
    sigs = [x.lower() for x in signatures if x]
    out = []
    for s in split_sentences(reply):
        s = s.strip()
        if not s or NON_CLAIM.match(s) or len(content_terms(tokenize(s))) < 3:
            continue
        if any(sig and (s.lower() == sig or s.lower() in sig or sig in s.lower() and len(s) < len(sig) + 15)
               for sig in sigs):
            continue  # sign-off such as "- Remya Industries Ltd Customer Support"
        out.append(s)
    return out


def support_ratio(reply: str, evidence: str, min_overlap: float = 0.5, signatures: List[str] = ()) -> Dict[str, Any]:
    """Share of factual reply sentences whose key words (>= min_overlap) appear in the evidence."""
    evidence_terms = set(tokenize(evidence))
    claims = _claim_sentences(reply, signatures)
    unsupported = []
    for s in claims:
        terms = set(content_terms(tokenize(s)))
        if terms and len(terms & evidence_terms) / len(terms) < min_overlap:
            unsupported.append(s)
    ratio = 1.0 if not claims else round(1 - len(unsupported) / len(claims), 3)
    return {"claims": len(claims), "supported_ratio": ratio, "unsupported_sentences": unsupported[:5]}


def check(reply: str, citations: List[str], refs: Dict[str, Dict[str, Any]], passages: List[Dict[str, Any]],
          customer_text: str = "", ids: List[str] = (), answer_found: bool = True,
          signatures: List[str] = ()) -> Dict[str, Any]:
    """Validate one reply. ``refs``: ref -> passage info for every passage retrieved for this ticket."""
    evidence = " ".join(p.get("text", "") for p in passages)
    valid = [c for c in dict.fromkeys(citations or []) if c in refs]
    invalid = [c for c in dict.fromkeys(citations or []) if c not in refs]
    allowed = set(numbers(" ".join([evidence, customer_text, *ids])))
    unsupported_figures = [n for n in dict.fromkeys(numbers(reply)) if n not in allowed]
    support = support_ratio(reply, evidence, signatures=signatures) if evidence else {"claims": 0, "supported_ratio": 1.0,
                                                               "unsupported_sentences": []}
    grounded = not unsupported_figures and (not answer_found or bool(valid) or not refs)
    if unsupported_figures:
        note = f"figures not found in the documents: {', '.join(unsupported_figures[:5])}"
    elif answer_found and refs and not valid:
        note = "no valid citation"
    else:
        note = "grounded"
    return {
        "grounded": grounded,
        "note": note,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "cited": [{"ref": c, **{k: refs[c].get(k, "") for k in ("source", "section", "chunk_id")}} for c in valid],
        "unsupported_figures": unsupported_figures,
        **support,
    }


def describe(grounding: Dict[str, Any]) -> str:
    """One-line summary for the ticket log."""
    if not grounding:
        return ""
    cites = ", ".join(f"{c['ref']} {c['source']}" for c in grounding.get("cited", []))
    return f"{grounding.get('note', '')}; support {grounding.get('supported_ratio', 1.0):.0%}" + (
        f"; cited {cites}" if cites else "")


# ------------------------------------------------------------------ LLM claim verifier
VERIFIER_SYSTEM = """You verify a customer-support reply against the company passages it was based on.
List every FACTUAL claim in the reply (steps, timelines, amounts, conditions, eligibility, promises) that the passages
do NOT support. A claim is supported when a passage states the same thing, even in different words.
Ignore greetings, empathy, the ticket number, "a specialist will contact you / follow up", and honest statements that
the information is not available. Do not judge style.
Answer with ONLY a JSON object: {"unsupported_claims": ["claim as written in the reply", ...]}"""


def verifier_enabled(settings) -> bool:
    mode = (getattr(settings, "grounding_verifier", "auto") or "auto").lower()
    return mode == "llm" or (mode == "auto" and not settings.is_offline)


async def llm_verify(client, reply: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Second, semantic grounding check: an LLM lists reply claims the retrieved passages do not support."""
    from autogen_core.models import SystemMessage, UserMessage

    from engine.parsing import extract_json

    evidence = "\n\n".join(f"[{p.get('ref') or i + 1}] {p.get('source', '')} > {p.get('section', '')}\n{p.get('text', '')}"
                           for i, p in enumerate(passages[:8])) or "(no passages)"
    try:
        result = await client.create([SystemMessage(content=VERIFIER_SYSTEM),
                                      UserMessage(content=f"PASSAGES:\n{evidence}\n\nREPLY:\n{reply}", source="verifier")])
        obj = extract_json(result.content if isinstance(result.content, str) else str(result.content)) or {}
        claims = [str(c).strip() for c in obj.get("unsupported_claims", []) if str(c).strip()]
        return {"ran": True, "unsupported_claims": claims[:6]}
    except Exception as exc:  # the verifier must never block a ticket
        return {"ran": False, "unsupported_claims": [], "error": f"{type(exc).__name__}: {exc}"[:200]}


def merge_llm_verdict(grounding: Dict[str, Any], verdict: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the LLM verifier's result into the grounding record (the reply is ungrounded if it found claims)."""
    g = dict(grounding or {})
    g["llm_verifier"] = verdict
    claims = verdict.get("unsupported_claims", []) if verdict.get("ran") else []
    if claims:
        g["grounded"] = False
        g["unsupported_claims"] = claims
        note = f"LLM verifier: {len(claims)} unsupported claim(s)"
        g["note"] = note if g.get("note") in {None, "", "grounded"} else f"{g['note']}; {note}"
    return g
