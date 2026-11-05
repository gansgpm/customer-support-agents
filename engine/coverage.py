"""Does the knowledge base really answer this customer? - the independent coverage check used in production.

Word overlap and similarity scores (``rag.knowledge.is_covered``) are a fast first estimate, but they cannot tell
"gift cards as a payment method" from "selling branded gift cards to a company", or "technician behaviour" from
"delivery staff behaviour". With an LLM configured, every ``search_kb`` call is therefore checked by a separate,
narrow LLM call that sees only the customer's message and the top passages and answers covered / not covered.
That verdict becomes the search's ``match_found``, the Knowledge Base Agent may lower it but never raise it, and the
Log Agent enforces it. ``evaluation/retrieval_eval.py`` measures exactly this prompt
("+ LLM rewrite + LLM coverage verdict (production)"), so the evaluated path is the path that runs.
"""
from __future__ import annotations

from typing import Any, Dict, List

VERDICT_PROMPT = """You are the Knowledge Base Agent of a customer-support desk. Decide whether the PASSAGES (company
policy documents) contain the answer to the CUSTOMER'S MESSAGE. Loosely related text does not count: the passages
must actually address what the customer asks. Answer with ONLY a JSON object: {"covered": true or false}"""

TOP_PASSAGES = 4


def verdict_enabled(settings) -> bool:
    mode = (getattr(settings, "coverage_verdict", "auto") or "auto").lower()
    return mode == "llm" or (mode == "auto" and not settings.is_offline)


def evidence(passages: List[Dict[str, Any]], n: int = TOP_PASSAGES) -> str:
    return "\n\n".join(f"[{i + 1}] {p['source']} > {p['section']}\n{p['text']}" for i, p in enumerate(passages[:n]))


def user_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    return f"CUSTOMER'S MESSAGE:\n{question}\n\nPASSAGES:\n{evidence(passages) or '(none)'}"


def parse(raw: str) -> bool:
    from engine.parsing import extract_json

    obj = extract_json(raw) or {}
    value = obj.get("covered", "true" in (raw or "").lower())
    return value.strip().lower() == "true" if isinstance(value, str) else bool(value)


async def llm_covered(client, question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """{"ran": True, "covered": bool} - or {"ran": False, ...} on an error (the estimate then stands)."""
    from autogen_core.models import SystemMessage, UserMessage

    if not passages:
        return {"ran": True, "covered": False}
    try:
        result = await client.create([SystemMessage(content=VERDICT_PROMPT),
                                      UserMessage(content=user_prompt(question, passages), source="coverage_check")])
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"ran": True, "covered": parse(raw)}
    except Exception as exc:  # never block a ticket
        return {"ran": False, "covered": None, "error": f"{type(exc).__name__}: {exc}"[:200]}
