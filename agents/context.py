"""Everything an agent's tools need for one ticket, bundled together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from engine.config import Settings
from engine.state import TicketState
from rag.knowledge import KnowledgeStore
from ticketing.ticket_log import TicketLog


@dataclass
class AgentContext:
    settings: Settings
    state: TicketState
    store: KnowledgeStore
    ticket_log: TicketLog
    dispatcher: Optional[Any] = None  # channels.dispatcher.Dispatcher (None = no outbound delivery)
    model_name: str = ""


def agent_name(division: str, role: str) -> str:
    """Unique AutoGen name per division group, e.g. electronics_query_agent."""
    return f"{division}_{role}"


def division_brief(ctx: AgentContext, division: str) -> str:
    """Appended to every group agent's system message so it knows which division it serves."""
    line = ctx.store.catalog.get(division)
    company = ctx.store.catalog.company or ctx.settings.company_name
    return (f"\n\nDIVISION: you work in the {line.name} agent group of {company} ({line.description}). "
            f"The Support Manager routed this ticket to your group. Read the Manager's brief (division, "
            f"secondary_divisions, guidance) first and stay within your division's policies plus the general "
            f"policies; when the ticket also involves a secondary division, cover that part too.")
