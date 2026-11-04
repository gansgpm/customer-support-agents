"""Answer faithfulness: an LLM-as-judge score plus a deterministic lexical score for every reply.

* **Lexical support** (always, no API key): share of the reply's factual sentences whose key words appear in
  the retrieved passages, plus the figure check (every number in the reply must be in the passages) -
  computed by engine/grounding.py, the same check that runs on every live ticket.
* **LLM judge** (when a real LLM is configured): the model reads the customer's message, the retrieved
  passages and the reply, and scores *faithfulness* (1-5: is every factual statement supported by the
  passages?) and *relevance* (1-5: does it answer what the customer asked?), listing unsupported claims.
  A judge model sees only the evidence, not the agents' reasoning, so it does not grade its own work.
"""
from __future__ import annotations

import json
from typing import Dict, List

from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from engine.parsing import extract_json

JUDGE_SYSTEM = """You are a strict quality auditor for customer-support replies.
Judge the REPLY only against the PASSAGES (the company documents retrieved for this ticket).
- faithfulness 1-5: 5 = every factual statement (steps, timelines, amounts, conditions, promises) is supported by the passages; 3 = mostly supported with minor additions; 1 = important claims are not in the passages.
  Greetings, empathy, the ticket number and "a specialist will contact you" are not factual claims.
  An honest "we don't have this information, a specialist will follow up" is fully faithful.
- relevance 1-5: 5 = fully answers what the customer asked; 1 = does not address it.
Answer with ONLY a JSON object: {"faithfulness": int, "relevance": int, "unsupported_claims": ["..."], "comment": "one sentence"}"""


async def judge_reply(client: ChatCompletionClient, complaint: str, passages: List[Dict], reply: str) -> Dict:
    evidence = "\n\n".join(f"[{p.get('ref') or i + 1}] {p.get('source', '')} > {p.get('section', '')}\n{p.get('text', '')}"
                           for i, p in enumerate(passages[:6])) or "(no passages were retrieved)"
    prompt = (f"CUSTOMER MESSAGE:\n{complaint}\n\nPASSAGES:\n{evidence}\n\nREPLY:\n{reply}\n\n"
              "Return the JSON object.")
    try:
        result = await client.create([SystemMessage(content=JUDGE_SYSTEM), UserMessage(content=prompt, source="judge")])
        obj = extract_json(result.content if isinstance(result.content, str) else json.dumps(result.content)) or {}
        return {"faithfulness": int(obj.get("faithfulness", 0)), "relevance": int(obj.get("relevance", 0)),
                "unsupported_claims": obj.get("unsupported_claims", []) or [],
                "comment": str(obj.get("comment", ""))[:200]}
    except Exception as exc:  # the judge is optional - never fail the evaluation because of it
        return {"faithfulness": None, "relevance": None, "unsupported_claims": [], "comment": f"judge error: {exc}"}
