"""Offline demo model client (no API key, no internet).

A deterministic, rule-based stand-in for an LLM that speaks AutoGen's
``ChatCompletionClient`` protocol: it emits real tool calls and JSON answers,
so the *same* agents, tools, team, termination, logging and UI run end to end.

Use it for: demos when the API quota runs out, CI / unit tests, and checking
the plumbing. It is not a language model — it does not translate and its
replies are template-based. Select it with ``LLM_PROVIDER=offline``.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from engine.escalation import HIGH_URGENCY_RISKS, RISK_LABELS, compute_priority
from engine.parsing import extract_json
from engine.state import MANAGER, role_of
from rag.text import split_sentences, tokenize

PIPELINE = ["query_agent", "knowledge_agent", "resolution_agent", "log_agent"]

CATEGORY_KEYWORDS = {
    "General": ["gift card", "franchise", "partnership", "bulk order", "corporate", "wholesale", "dealership"],
    "Service": ["installation", "install", "technician", "repair", "remyacare", "remyashield", "claim",
                "protection plan", "extended warranty", "membership", "remya plus", "renewal", "subscription"],
    "Account": ["password", "log in", "login", "logged in", "logged into", "otp", "account", "locked", "hacked", "sign in"],
    "Billing": ["charged", "charge", "payment", "card", "invoice", "promo", "coupon", "code", "bill", "upi", "emi", "debited"],
    "Refund": ["refund", "return", "exchange", "cancel", "money back"],
    "Product": ["broken", "defective", "damaged product", "not working", "replacement", "warranty", "faulty", "stopped working",
                "battery", "overheat", "swelling", "not switching on", "spoiled", "rotten", "expired", "stale", "sole",
                "insect", "foreign object", "worm", "fungus", "sour", "smells", "leaking", "stinks"],
    "Shipping": ["deliver", "delivery", "arrive", "track", "late", "shipped", "courier", "package", "parcel", "missing",
                 "torn", "opened", "dispatch", "where is my order", "not received", "ship"],
}
PROBLEM_CUES = ["late", "broken", "missing", "not working", "can't", "cannot", "forgot", "failed", "damaged", "torn",
                "did not come", "didn't come", "no-show", "pending", "spoiled", "rotten", "expired", "rejected",
                "not switching on", "wrong size",
                "invalid", "isn't", "wrong", "stuck", "supposed to", "where is my order", "not received", "delay",
                "never", "didn't", "locked"]
INTENTS = {
    "Shipping": "get an update on / fix a delivery problem",
    "Refund": "return an item and get a refund",
    "Account": "regain secure access to the account",
    "Product": "get a damaged or defective product replaced",
    "Billing": "resolve a payment or billing problem",
    "Service": "get an installation, repair, protection-plan or membership issue sorted",
    "General": "get information that is outside the standard policies",
}
EMPATHY = {
    "Angry": "I'm truly sorry for the trouble this has caused, and I understand why you are upset.",
    "Negative": "I'm sorry for the inconvenience this has caused.",
    "Neutral": "Thank you for reaching out to us.",
    "Positive": "Thank you for reaching out to us.",
}


def _usage() -> RequestUsage:
    return RequestUsage(prompt_tokens=0, completion_tokens=0)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


class OfflineChatCompletionClient(ChatCompletionClient):
    component_type = "model"

    def __init__(self) -> None:
        self._total = _usage()
        self._info: ModelInfo = {"vision": False, "function_calling": True, "json_output": True,
                                 "family": ModelFamily.UNKNOWN, "structured_output": False}

    # ------------------------------------------------------------------ API
    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = [],
                     tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
                     json_output: Optional[bool | type[BaseModel]] = None, extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        content = self._respond(list(messages), [self._tool_name(t) for t in tools])
        return CreateResult(finish_reason="function_calls" if isinstance(content, list) else "stop",
                            content=content, usage=_usage(), cached=False)

    async def create_stream(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = [],
                            tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
                            json_output: Optional[bool | type[BaseModel]] = None,
                            extra_create_args: Mapping[str, Any] = {},
                            cancellation_token: Optional[CancellationToken] = None
                            ) -> AsyncGenerator[Union[str, CreateResult], None]:
        result = await self.create(messages, tools=tools)
        if isinstance(result.content, str):  # stream in small pieces, like a real model, so the UI path is tested
            for i in range(0, len(result.content), 48):
                yield result.content[i:i + 48]
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return _usage()

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return sum(len(_text(getattr(m, "content", "")).split()) for m in messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 100_000

    @property
    def capabilities(self):  # type: ignore[override]
        return self._info

    @property
    def model_info(self) -> ModelInfo:
        return self._info

    # ------------------------------------------------------------- routing
    @staticmethod
    def _tool_name(tool: Tool | ToolSchema) -> str:
        return tool.name if hasattr(tool, "name") else tool["name"]  # type: ignore[index]

    def _respond(self, messages: List[LLMMessage], tool_names: List[str]) -> Union[str, List[FunctionCall]]:
        system = " ".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
        all_text = " ".join(_text(getattr(m, "content", "")) for m in messages)
        if "select the next role" in all_text.lower() or "select the next" in system.lower():
            return self._select_next(all_text)

        ticket = self._ticket(messages)
        outputs = {role_of(m.source): _text(m.content) for m in messages
                   if isinstance(m, UserMessage) and role_of(m.source)}
        tool_results = self._own_tool_results(messages)

        if system.startswith("You verify a customer-support reply"):  # grounding.llm_verify
            return self._verify(all_text)
        if "You are the Support Manager" in system:
            if not tool_results and tool_names:
                return [self._call("identify_product_line", {"text": ticket["complaint"]})]
            return self._route(ticket, tool_results)
        if "You are the Customer Query Agent" in system:
            if not tool_results and tool_names:
                return [self._call("analyze_sentiment", {"text": ticket["complaint"]}),
                        self._call("search_ticket_history", {"query": ticket["complaint"]})]
            return self._triage(ticket, tool_results, extract_json(outputs.get(MANAGER, "")) or {})
        if "You are the Knowledge Base Agent" in system:
            triage = extract_json(outputs.get("query_agent", "")) or {}
            if not tool_results and tool_names:
                query = triage.get("standalone_query") or f"{triage.get('summary', '')} {ticket['complaint']}".strip()
                lines = triage.get("product_lines") or [triage.get("product_line", "general")]
                return [self._call("search_kb", {"query": query, "category": triage.get("category", ""),
                                                 "product_line": ",".join(lines)})]
            return self._kb_pack(ticket, triage, tool_results)
        if "You are the Resolution Agent" in system:
            triage = extract_json(outputs.get("query_agent", "")) or {}
            kb = extract_json(outputs.get("knowledge_agent", "")) or {}
            return self._resolve(ticket, triage, kb)
        if "You are the Log Agent" in system:
            triage = extract_json(outputs.get("query_agent", "")) or {}
            kb = extract_json(outputs.get("knowledge_agent", "")) or {}
            res = extract_json(outputs.get("resolution_agent", "")) or {}
            reasons = self._log_reasons(triage, kb, res, system)
            return [self._call("log_ticket", {"escalate": bool(reasons), "escalation_reason": ", ".join(reasons)})]
        return "I am an offline demo model."

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _call(name: str, args: Dict) -> FunctionCall:
        return FunctionCall(id=f"call_{uuid.uuid4().hex[:10]}", name=name, arguments=json.dumps(args))

    @staticmethod
    def _ticket(messages: List[LLMMessage]) -> Dict[str, str]:
        task = next((_text(m.content) for m in messages if isinstance(m, UserMessage) and m.source == "user"), "")
        get = lambda pat: (re.search(pat, task, flags=re.S | re.I) or [None, ""])[1].strip()  # noqa: E731
        name = get(r"Customer name:\s*(.*?)\n")
        previous = get(r"Previous request in this conversation:\s*(.*?)\n")
        return {
            "previous": previous,
            "ticket_id": get(r"NEW SUPPORT TICKET\s+(TKT-\d+)"),
            "name": "" if name.lower().startswith("not provided") else name,
            "complaint": get(r'Complaint[^\n]*:\s*"""\s*(.*?)\s*"""') or task,
            "guard_flags": get(r"Guardrail flags:\s*(.*?)\n"),
        }

    @staticmethod
    def _own_tool_results(messages: List[LLMMessage]) -> Dict[str, Any]:
        """Tool results produced in the agent's current turn (after its last function-call message)."""
        results: Dict[str, Any] = {}
        for m in reversed(messages):
            if isinstance(m, FunctionExecutionResultMessage):
                for r in m.content:
                    try:
                        results[r.name] = json.loads(r.content)
                    except (json.JSONDecodeError, TypeError):
                        results[r.name] = r.content
            elif isinstance(m, AssistantMessage) and isinstance(m.content, list):
                continue
            else:
                break
        return results

    @staticmethod
    def _select_next(prompt: str) -> str:
        """Pick the candidate (from the selector prompt's list) that comes earliest in the pipeline."""
        lists = re.findall(r"\[([^\[\]]*)\]", prompt)
        names = re.findall(r"'([^']+)'", lists[-1]) if lists else []
        order = [MANAGER] + PIPELINE
        ranked = sorted(names, key=lambda n: order.index(role_of(n)) if role_of(n) in order else 99)
        return ranked[0] if ranked else PIPELINE[0]

    def _route(self, ticket: Dict, tools: Dict) -> str:
        """Support Manager: owning division = the tool's primary match (services win ties), rest = secondary.

        Not a support request (no business words, nothing detected, knowledge base silent) -> ROUTE: none.
        A real issue that no division or document covers -> general group with needs_human.
        """
        from engine.guardrail import SUPPORT_CUES

        res = tools.get("identify_product_line", {}) if isinstance(tools.get("identify_product_line"), dict) else {}
        kb = res.get("kb_check", {})
        risks = [r["flag"] for r in res.get("line_risks", [])]
        if res.get("unroutable") and not SUPPORT_CUES.search(ticket["complaint"]):
            out = {"ticket_type": "not_support", "division": "general", "secondary_divisions": [],
                   "product_or_service": "", "route_confidence": "high", "needs_human": False,
                   "guardrail_reason": "off_topic",
                   "reason": "The message is not about Remya Industries Ltd orders, products or services.",
                   "guidance": "", "line_risks": []}
            return "```json\n" + json.dumps(out, indent=2) + "\n```\nROUTE: none"
        division = res.get("primary", "general")
        secondary = [d for d in res.get("all", []) if d not in {division, "general"}]
        matched = [m for d in res.get("detected", []) for m in d.get("matched", [])[:1]]
        needs_human = bool(res.get("unroutable"))
        if matched:
            why, confidence = f"the message mentions {', '.join(matched)}", "high"
        elif res.get("routed_by") == "knowledge-base match":
            why, confidence = f"the knowledge base points to {kb.get('best_guide', '')}", "medium"
        elif needs_human:
            why, confidence = "no division or company document covers it, so a human specialist must decide", "low"
        else:
            why, confidence = "no product line or service is mentioned, so the general group owns it", "high"
        out = {"ticket_type": "support_issue", "division": division, "secondary_divisions": secondary,
               "product_or_service": ", ".join(matched[:2]), "route_confidence": confidence,
               "needs_human": needs_human, "guardrail_reason": "",
               "reason": f"Routed to {res.get('primary_name', division)}: {why}.",
               "guidance": ("Watch the line-specific risk: " + ", ".join(risks) + ". " if risks else "")
               + f"Use the {res.get('primary_name', division)} guide plus general policies.",
               "line_risks": risks}
        return "```json\n" + json.dumps(out, indent=2, ensure_ascii=False) + f"\n```\nROUTE: {division}"

    # ---------------------------------------------------------------- agents
    def _triage(self, ticket: Dict, tools: Dict, route: Dict) -> str:
        text = ticket["complaint"]
        lower = text.lower()
        signal = tools.get("analyze_sentiment", {}) if isinstance(tools.get("analyze_sentiment"), dict) else {}
        history = tools.get("search_ticket_history", {}) if isinstance(tools.get("search_ticket_history"), dict) else {}

        scores = {cat: sum(1 for k in kws if k in lower) for cat, kws in CATEGORY_KEYWORDS.items()}
        if scores["General"]:
            category = "General"
        else:
            best = max(scores.values())
            category = next((c for c in ["Account", "Service", "Billing", "Product", "Refund", "Shipping"]
                             if scores[c] == best), "General") if best else "General"

        risk = sorted(set(signal.get("risk_flags", [])) | set(route.get("line_risks", []))
                      | ({"prompt_injection"} if "prompt_injection" in ticket.get("guard_flags", "") else set()))
        sentiment = signal.get("sentiment", "Neutral")
        if any(r in HIGH_URGENCY_RISKS for r in risk):
            urgency, why = "High", "risk flag: " + ", ".join(risk)
        elif sentiment == "Angry" or signal.get("urgency_hint") == "High":
            urgency, why = "High", "very angry customer or long overdue with unanswered contact"
        elif any(c in lower for c in PROBLEM_CUES):
            urgency, why = "Medium", "a paid order or service is not working"
        else:
            urgency, why = "Low", "information request"

        repeat = bool(history.get("same_customer_or_order"))
        words = re.sub(r"\s+", " ", text).split()
        summary = " ".join(words[:22]) + ("..." if len(words) > 22 else "")
        triage = {
            "category": category, "urgency": urgency, "urgency_reason": why, "sentiment": sentiment,
            "sentiment_score": signal.get("sentiment_score", 0.0), "language": signal.get("language", "en"),
            "product_line": route.get("division", "general"),
            "product_lines": [route.get("division", "general")] + route.get("secondary_divisions", []),
            "product_or_service": route.get("product_or_service", ""),
            "summary": f"{category} issue: {summary}", "intent": INTENTS[category], "risk_flags": risk,
            "standalone_query": (f"{re.sub(r'^[A-Za-z]+ issue:\s*', '', ticket['previous'])} - follow-up: {text}"
                                 if ticket.get("previous") else ""),
            "repeat_issue": repeat,
            "similar_tickets": [t["ticket_id"] for t in history.get("similar_tickets", [])][:3],
        }
        triage["priority_score"] = compute_priority(triage)
        return "```json\n" + json.dumps(triage, indent=2, ensure_ascii=False) + "\n```"

    def _kb_pack(self, ticket: Dict, triage: Dict, tools: Dict) -> str:
        res = tools.get("search_kb", {}) if isinstance(tools.get("search_kb"), dict) else {}
        q_tokens = set(tokenize(f"{triage.get('summary', '')} {ticket['complaint']}"))
        all_passages = res.get("policy_passages", []) if res.get("policy_match") else []
        passages = all_passages[:2]
        facts: List[Dict[str, str]] = []

        def usable(s: str) -> bool:
            return len(s.split()) >= 3 and not s.rstrip().endswith(":")

        lines = [l for l in triage.get("product_lines", []) if l and l != "general"]
        per_line = []
        if len(lines) > 1:  # multi-line ticket: best passage for each product line / service
            for line in lines[:3]:
                best = next((p for p in all_passages if p.get("scope") == line), None)
                if best:
                    per_line.append(best)
        if len(per_line) > 1:
            for p in per_line:
                facts += [{"ref": p.get("ref", ""), "source": p["source"], "fact": s}
                          for s in split_sentences(p["text"]) if usable(s)][:3]
        elif passages:
            # the best-ranked passage (search_kb already reranked them), kept whole so steps stay coherent
            top = passages[0]
            second = passages[1] if len(passages) > 1 else None
            close = bool(second) and second.get("rank_score", 0) >= 0.85 * top.get("rank_score", 1)
            facts += [{"ref": top.get("ref", ""), "source": top["source"], "fact": s}
                      for s in split_sentences(top["text"]) if usable(s)][:4 if close else 6]
            if second:
                extra = [s for s in split_sentences(second["text"])
                         if usable(s) and (close or len(q_tokens & set(tokenize(s))) >= 3)]
                facts += [{"ref": second.get("ref", ""), "source": second["source"], "fact": s}
                          for s in extra[:3 if close else 2]]
        learned = [{"ticket_id": m.get("ticket_id", ""), "note": m["text"][:160]}
                   for m in res.get("resolved_ticket_matches", [])]
        match = bool(res.get("match_found")) and bool(facts or learned)
        sources = sorted({f["source"] for f in facts})
        brief = (f"Customer needs to {triage.get('intent', 'get help')}. Relevant policy: "
                 f"{', '.join(sources) or 'none'}. Reply must include the facts below verbatim."
                 if match else "No company document covers this request; the reply must say so honestly.")
        pack = {"match_found": match, "sources": sources, "key_facts": facts[:9], "similar_resolutions": learned,
                "context_brief": brief, "gaps": "" if match else "Topic not covered by Remya Industries Ltd policies."}
        return "```json\n" + json.dumps(pack, indent=2, ensure_ascii=False) + "\n```"

    @staticmethod
    def _verify(text: str) -> str:
        """Offline stand-in for the LLM claim verifier: reply sentences whose key words are mostly absent from the
        passages (the lexical check; a real LLM also catches paraphrased or contradicted claims)."""
        from engine.grounding import support_ratio

        passages, _, reply = text.partition("REPLY:")
        claims = support_ratio(reply.strip(), passages.replace("PASSAGES:", ""))["unsupported_sentences"]
        return json.dumps({"unsupported_claims": claims})

    def _resolve(self, ticket: Dict, triage: Dict, kb: Dict) -> str:
        name = ticket["name"].split()[0] if ticket["name"] else "there"
        tid = ticket["ticket_id"]
        empathy = EMPATHY.get(triage.get("sentiment", "Neutral"), EMPATHY["Neutral"])
        facts = [f["fact"] for f in kb.get("key_facts", []) if isinstance(f, dict)]
        risky = any(r in RISK_LABELS for r in triage.get("risk_flags", []))
        if kb.get("match_found") and facts:
            body = "\n".join(f"- {f}" for f in facts[:9])
            closing = ("Because this involves your account or money, a Remya Industries Ltd specialist will also contact you "
                       "personally." if risky else "If anything is still unclear, just reply to this message.")
            response = (f"Hi {name},\n\n{empathy} Here is what applies to your request:\n{body}\n\n{closing} "
                        f"Your ticket number is {tid}.\n\n- Remya Industries Ltd Customer Support")
            confidence = 88 if len(facts) >= 2 else 74
            sources, found = kb.get("sources", []), True
        else:
            response = (f"Hi {name},\n\n{empathy} I'm sorry, but I don't have verified information about this request "
                        f"in our published policies, so I don't want to guess. I've passed ticket {tid} to a Remya Industries Ltd "
                        f"specialist who will follow up with you.\n\n- Remya Industries Ltd Customer Support")
            confidence, sources, found = 25, [], False
        cited = [f.get("ref") for f in kb.get("key_facts", []) if isinstance(f, dict) and f.get("ref")]
        out = {"response": response, "confidence": confidence, "used_sources": sources,
               "citations": list(dict.fromkeys(cited)) if found else [], "answer_found": found,
               "language": triage.get("language", "en")}
        return "```json\n" + json.dumps(out, indent=2, ensure_ascii=False) + "\n```"

    @staticmethod
    def _log_reasons(triage: Dict, kb: Dict, res: Dict, system: str) -> List[str]:
        m = re.search(r"confidence is below (\d+)", system)
        threshold = int(m.group(1)) if m else 80
        reasons = []
        if triage.get("urgency") == "High":
            reasons.append("High urgency")
        if triage.get("sentiment") == "Angry":
            reasons.append("Very angry customer")
        reasons += [f"Involves {RISK_LABELS[r]}" for r in triage.get("risk_flags", []) if r in RISK_LABELS]
        if not kb.get("match_found", False):
            reasons.append("No matching document")
        if int(res.get("confidence", 0) or 0) < threshold:
            reasons.append("Low confidence")
        return reasons
