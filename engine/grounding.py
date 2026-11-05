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
# "1. Open the app" / "2) Tap Login" - list numbering, not a figure the documents have to contain
ENUMERATOR = re.compile(r"(?m)^[ \t]*\(?\d{1,2}[.)](?=\s)")
# Sentences that are courtesy, not claims (greeting, empathy, ticket reference, sign-off)
NON_CLAIM = re.compile(r"^(hi|hello|dear|thank|thanks|sorry|i'?m sorry|we'?re sorry|i understand|"
                       r"your ticket number|ticket number|if anything is still unclear|best regards|regards|-\s)",
                       re.I)


def numbers(text: str) -> List[str]:
    return [n.replace(",", "") for n in NUMBER.findall(ENUMERATOR.sub(" ", text or ""))]


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
VERIFIER_SYSTEM = """You verify a customer-support reply against the company PASSAGES it was based on and the TICKET (the
customer's own message and ticket details).
Go through the reply and list each FACTUAL claim: steps, timelines, amounts, conditions, eligibility, promises of what
the company will do. For each claim decide:
- "supported": a passage states it (the same meaning in different words is fine), or it only repeats the TICKET (the
  customer's name, order id, ticket number, what the customer said happened);
- "unsupported": no passage states it, or a passage says something different.
Skip greetings, empathy, thanks, the ticket number, "a specialist will contact you / follow up", "contact support if you
need more help", and honest statements that the information is not available. Do not judge style.
Answer with ONLY a JSON object:
{"claims": [{"claim": "the claim as written in the reply", "verdict": "supported" or "unsupported",
             "evidence": "the passage or ticket sentence that supports it (empty when unsupported)"}]}"""

# How the desk itself works - a human follows up, the customer can write back - is not a policy claim the documents
# must contain. The verifier prompt says to skip these; this enforces it when the model flags one anyway.
DESK_PROCESS = re.compile(
    r"(\b(specialist|support team|support agent|customer care|our team|someone|we)\b[^.]{0,60}"
    r"\b(will|can|shall)\b[^.]{0,20}\b(follow(?:s|ing)? up|get back|contact you|reach out|be in touch|"
    r"keep you (?:informed|updated|posted)|check (?:the|your|on) (?:status|order|ticket)|look into)"
    r"|\b(let us know|feel free to|reply to this (?:message|email)|get in touch with us|reach out to us|"
    r"contact(?:ing)? (?:us|our support|our customer care|support|customer care)\b(?! to\b)))", re.I)


def is_desk_process(claim: str) -> bool:
    """'A support specialist will follow up', 'let us know and we can check the status' - true by how the desk
    works (escalated tickets go to a specialist, replies are linked to the ticket); never a statement with figures."""
    return bool(DESK_PROCESS.search(claim or "")) and not numbers(claim)


NEGATION = re.compile(r"\b(not|no|never|cannot|can't|won't|don't|doesn't|isn't|aren't|without|non)\b", re.I)
MAX_VERIFIER_PASSAGES = 12


def verifier_enabled(settings) -> bool:
    mode = (getattr(settings, "grounding_verifier", "auto") or "auto").lower()
    return mode == "llm" or (mode == "auto" and not settings.is_offline)


def near_verbatim(claim: str, sentences: List[str], threshold: float = 0.8) -> bool:
    """True when one source sentence states the claim almost word for word: >= 80% of its key words, every figure
    in it, and the same negation. Used to overrule a verifier that flags a claim copied from the documents."""
    terms = set(content_terms(tokenize(claim)))
    if len(terms) < 3:
        return False
    figures, negated = set(numbers(claim)), bool(NEGATION.search(claim))
    for sentence in sentences:
        words = set(tokenize(sentence))
        if (len(terms & words) / len(terms) >= threshold and figures <= set(numbers(sentence))
                and negated == bool(NEGATION.search(sentence))):
            return True
    return False


def _verifier_passages(passages: List[Dict[str, Any]], cited: List[str] = ()) -> List[Dict[str, Any]]:
    """Cited passages first, then the rest of what was retrieved (deduplicated, capped)."""
    cited = list(cited or [])
    order = sorted(range(len(passages)), key=lambda i: (passages[i].get("ref") not in cited, i))
    seen, out = set(), []
    for i in order:
        p = passages[i]
        key = (p.get("source"), p.get("section"), p.get("text", "")[:80])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out[:MAX_VERIFIER_PASSAGES]


async def llm_verify(client, reply: str, passages: List[Dict[str, Any]], customer_text: str = "",
                     ticket_details: str = "", cited: List[str] = ()) -> Dict[str, Any]:
    """Second, semantic grounding check: an LLM lists the reply's claims that neither the retrieved passages nor the
    ticket support. A flagged claim that a passage (or the ticket) states almost word for word is overruled."""
    from autogen_core.models import SystemMessage, UserMessage

    from engine.parsing import extract_json

    chosen = _verifier_passages(passages, cited)
    evidence = "\n\n".join(f"[{p.get('ref') or i + 1}] {p.get('source', '')} > {p.get('section', '')}\n{p.get('text', '')}"
                           for i, p in enumerate(chosen)) or "(no passages)"
    ticket = f"Customer's message: {customer_text.strip()}\n{ticket_details.strip()}".strip()
    try:
        result = await client.create([SystemMessage(content=VERIFIER_SYSTEM), UserMessage(
            content=f"PASSAGES:\n{evidence}\n\nTICKET:\n{ticket}\n\nREPLY:\n{reply}", source="verifier")])
        obj = extract_json(result.content if isinstance(result.content, str) else str(result.content)) or {}
    except Exception as exc:  # the verifier must never block a ticket
        return {"ran": False, "unsupported_claims": [], "error": f"{type(exc).__name__}: {exc}"[:200]}
    if isinstance(obj.get("claims"), list):
        flagged = [str(c.get("claim", "")).strip() for c in obj["claims"]
                   if isinstance(c, dict) and str(c.get("verdict", "")).lower().startswith("unsupported")]
    else:  # older / simpler answer format
        flagged = [str(c).strip() for c in obj.get("unsupported_claims", [])]
    flagged = [c for c in flagged if c]
    sources = [s for p in chosen for s in split_sentences(p.get("text", ""))]
    sources.append(f"{customer_text} {ticket_details}")  # the ticket as one block (name, order id, what happened)
    sources += split_sentences(customer_text)
    overruled = [c for c in flagged if is_desk_process(c) or near_verbatim(c, sources)]
    claims = [c for c in flagged if c not in overruled]
    return {"ran": True, "unsupported_claims": claims[:6], "overruled": overruled[:6],
            "checked_claims": len(obj["claims"]) if isinstance(obj.get("claims"), list) else None}


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
