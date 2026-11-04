"""Customer-facing message templates (the 'wrapped' template mail / chat message)."""
from __future__ import annotations

from typing import Dict

from engine.config import Settings
from engine.state import TicketState

# Response-time commitments from knowledge_base/support_and_escalation_policy.md
SLA_BY_URGENCY = {
    "High": "2 hours (support hours 9:00 AM - 9:00 PM IST)",
    "Medium": "8 hours",
    "Low": "24 hours",
}


def holding_message(state: TicketState, decision: Dict, settings: Settings, line_sla: str = "") -> str:
    name = state.ticket.customer_name.strip() or "there"
    urgency = state.triage.get("urgency", "Medium")
    sla = line_sla or SLA_BY_URGENCY.get(urgency, SLA_BY_URGENCY["Medium"])
    team = state.routing.get("name", "")
    desk = f"our {team}" if team else "a human support specialist"
    return (
        f"Hi {name}, thank you for contacting {settings.company_name}. "
        f"Your request has been registered as ticket {state.ticket_id} and handed to {desk}, "
        f"where a specialist will review it personally. You can expect a response on this channel within {sla}. "
        f"Please keep your ticket number for reference.\n\n- {settings.support_signature}"
    )


def customer_message(state: TicketState, decision: Dict, settings: Settings, line_sla: str = "") -> Dict[str, str]:
    """What the customer actually receives on their channel."""
    reply = state.resolution.get("response", "").strip()
    if decision["status"] == "Escalated" and settings.escalated_reply_mode == "holding":
        return {"kind": "holding", "text": holding_message(state, decision, settings, line_sla)}
    if decision["status"] == "Escalated":
        note = f"\n\n(Ticket {state.ticket_id} has also been passed to a human specialist for follow-up.)"
        return {"kind": "reply+escalation_notice", "text": reply + note}
    return {"kind": "resolution", "text": reply}


def support_alert(state: TicketState, decision: Dict) -> str:
    t = state.triage
    return (
        f"ESCALATED {state.ticket_id} -> {state.routing.get('name', 'Support')} "
        f"[{t.get('urgency', '?')} | {t.get('category', '?')} | {state.routing.get('product_line_name', '')} | "
        f"{t.get('sentiment', '?')}]\n"
        f"Channel: {state.ticket.channel} - {state.ticket.customer_name or 'Unknown'} {state.ticket.customer_contact}\n"
        f"Order: {state.ticket.order_id or '-'}\n"
        f"Reasons: {', '.join(decision['reasons'])}\n"
        f"Complaint: {state.ticket.complaint[:400]}\n"
        f"Draft reply awaiting approval in the Escalation Queue."
    )
