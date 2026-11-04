"""Per-ticket shared state.

Tools write their raw results here (deterministic, race-free) and the
orchestrator stores each agent's parsed output here as it streams in.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

MANAGER = "support_manager"
GROUP_ROLES = ["query_agent", "knowledge_agent", "resolution_agent", "log_agent"]
AGENT_ORDER = [MANAGER] + GROUP_ROLES  # the order every ticket should follow
AGENT_TITLES = {
    MANAGER: "Support Manager",
    "query_agent": "Customer Query Agent",
    "knowledge_agent": "Knowledge Base Agent",
    "resolution_agent": "Resolution Agent",
    "log_agent": "Log Agent",
}


def role_of(agent_name: str) -> str:
    """'electronics_query_agent' -> 'query_agent'; 'support_manager' -> 'support_manager'; else ''."""
    if agent_name == MANAGER:
        return MANAGER
    for role in GROUP_ROLES:
        if agent_name == role or agent_name.endswith("_" + role):
            return role
    return ""


def division_of(agent_name: str) -> str:
    role = role_of(agent_name)
    if role in GROUP_ROLES and agent_name.endswith("_" + role):
        return agent_name[: -len(role) - 1]
    return ""


@dataclass
class TicketInput:
    complaint: str
    customer_name: str = ""
    order_id: str = ""
    channel: str = "web"  # web | email | telegram | whatsapp
    customer_contact: str = ""  # email address / telegram chat id / whatsapp number
    subject: str = ""
    inbox_id: str = ""  # link back to the channel inbox item
    reply_channels: List[str] = field(default_factory=list)  # extra channels to send the reply on
    reply_contacts: Dict[str, str] = field(default_factory=dict)
    product_line_hint: str = ""  # product line / service chosen by the customer (web form), "" = auto-detect
    parent_ticket: str = ""  # follow-up to this earlier ticket ("" = new conversation; detected automatically)


@dataclass
class TicketState:
    ticket_id: str
    ticket: TicketInput
    received_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    started: float = field(default_factory=time.time)

    # tool outputs (raw)
    sentiment_signal: Dict[str, Any] = field(default_factory=dict)
    product_signal: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    kb_results: List[Dict[str, Any]] = field(default_factory=list)

    # guardrails
    guardrail: Dict[str, Any] = field(default_factory=dict)  # input guardrail result (allowed, reason, flags)
    original_length: int = 0

    # parsed agent outputs
    manager_route: Dict[str, Any] = field(default_factory=dict)  # Support Manager's routing decision
    division: str = ""  # agent group that handled the ticket
    triage: Dict[str, Any] = field(default_factory=dict)
    kb_context: Dict[str, Any] = field(default_factory=dict)
    resolution: Dict[str, Any] = field(default_factory=dict)

    # outcome
    thread: Dict[str, Any] = field(default_factory=dict)  # earlier turns when this message is a follow-up
    citation_refs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # "S1" -> retrieved passage
    grounding: Dict[str, Any] = field(default_factory=dict)  # post-generation check of the reply
    escalation: Dict[str, Any] = field(default_factory=dict)
    routing: Dict[str, Any] = field(default_factory=dict)  # owning product line + specialist desk
    log_row: Dict[str, Any] = field(default_factory=dict)
    kb_feed: Dict[str, Any] = field(default_factory=dict)
    logged_by: str = ""  # "log_agent" or "engine_fallback"

    # trace
    speaker_order: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_outputs: Dict[str, str] = field(default_factory=dict)
    parse_errors: Dict[str, str] = field(default_factory=dict)
    agent_seconds: Dict[str, float] = field(default_factory=dict)
    stop_reason: str = ""
    message_count: int = 0
    llm_usage: Dict[str, int] = field(default_factory=dict)

    _events: Dict[str, asyncio.Event] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- events
    def event(self, agent: str) -> asyncio.Event:
        if agent not in self._events:
            self._events[agent] = asyncio.Event()
        return self._events[agent]

    async def wait_for(self, agent: str, timeout: float = 20.0) -> bool:
        try:
            await asyncio.wait_for(self.event(agent).wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # --------------------------------------------------------------- helpers
    def assign_ref(self, passage: Dict[str, Any]) -> str:
        """Stable citation number for a retrieved passage: the same chunk keeps the same ref across searches."""
        key = passage.get("chunk_id") or f"{passage.get('source')}#{passage.get('section')}"
        for ref, p in self.citation_refs.items():
            if p.get("_key") == key:
                return ref
        ref = f"S{len(self.citation_refs) + 1}"
        self.citation_refs[ref] = {"_key": key, "source": passage.get("source", ""),
                                   "section": passage.get("section", ""), "chunk_id": passage.get("chunk_id", ""),
                                   "text": passage.get("text", "")}
        return ref

    @property
    def kb_merged(self) -> Dict[str, Any]:
        """All search_kb calls merged (an agent may search more than once)."""
        merged: Dict[str, Any] = {"match_found": False, "policy_match": False, "learned_match": False,
                                  "top_relevance": 0.0, "top_coverage": 0.0, "policy_passages": [],
                                  "resolved_ticket_matches": []}
        seen = set()
        for res in self.kb_results:
            for key in ("match_found", "policy_match", "learned_match"):
                merged[key] = merged[key] or bool(res.get(key))
            merged["top_relevance"] = max(merged["top_relevance"], res.get("top_relevance", 0.0))
            merged["top_coverage"] = max(merged["top_coverage"], res.get("top_coverage", 0.0))
            for p in res.get("policy_passages", []):
                key = (p["source"], p["section"])
                if key not in seen:
                    seen.add(key)
                    merged["policy_passages"].append(p)
            for p in res.get("resolved_ticket_matches", []):
                if p not in merged["resolved_ticket_matches"]:
                    merged["resolved_ticket_matches"].append(p)
        return merged

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.started, 2)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("_events", None)
        data["kb_merged"] = self.kb_merged
        return data


@dataclass
class TicketResult:
    ticket_id: str
    status: str
    reply: str
    customer_message: str
    state: TicketState
    dispatch: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
