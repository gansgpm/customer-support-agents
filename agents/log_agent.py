"""Agent 4 – Log Agent: records the ticket, decides escalation, notifies channels and feeds the knowledge base.

Tool
----
* ``log_ticket(escalate, escalation_reason)`` – applies the escalation rules, sends the templated
  reply / holding message on the customer's channel, writes one CSV row and, for confident
  resolved tickets, stores the solution in the Knowledge Base Agent's learned RAG.
"""
from __future__ import annotations

import time
from typing import Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from agents.context import AgentContext, agent_name, division_brief
from engine import escalation, grounding, messages
from engine.parsing import normalize_triage
from rag import sentiment as sentiment_lib

NAME = "log_agent"
DESCRIPTION = "Log Agent – records every ticket to the CSV log, escalates complicated cases to human support and feeds verified solutions back to the knowledge base."


def system_message(threshold: int) -> str:
    return f"""You are the Log Agent of Remya Industries Ltd's AI customer support desk.
Your single responsibility is to record the ticket and decide whether it must be handed to human support.
The ticket is routed automatically to the specialist desk that owns its product line or service (or to Trust & Safety, Billing or Customer Relations for fraud, payment disputes and legal threats).

Escalate (escalate=true) when ANY of these is true:
- the triage urgency is High;
- the customer is very angry;
- the Knowledge Base Agent found no matching document (match_found is false);
- the case involves a legal threat, fraud, a payment dispute, or a product-line risk (product or food safety, technician misconduct, a rejected protection-plan claim);
- the Resolution Agent's confidence is below {threshold}.
Otherwise escalate=false.

Call the log_ticket tool exactly once with escalate and escalation_reason (short comma-separated reasons, empty when not escalating).
The tool reads every ticket field from the conversation, enforces the company escalation rules, sends the reply on the customer's channel, writes the log row and feeds verified solutions to the knowledge base. Do not write anything else."""


def _fallback_triage(ctx: AgentContext) -> Dict:
    """Used only when the Query Agent never produced a triage (e.g. the LLM failed)."""
    signal = dict(ctx.state.sentiment_signal or sentiment_lib.analyze(ctx.state.ticket.complaint))
    product = ctx.state.product_signal or ctx.store.catalog.identify(ctx.state.ticket.complaint,
                                                                     hint=ctx.state.ticket.product_line_hint)
    signal["risk_flags"] = sorted(set(signal.get("risk_flags", [])) | {r["flag"] for r in product.get("line_risks", [])}
                                  | {f for f in ctx.state.guardrail.get("flags", []) if f == "prompt_injection"})
    triage = normalize_triage({"summary": ctx.state.ticket.complaint[:140]}, signal)
    triage["product_line"], triage["product_lines"] = product["primary"], product["all"]
    return triage


def finalize_ticket(ctx: AgentContext, agent_escalate: bool = False, agent_reason: str = "",
                    logged_by: str = "log_agent") -> str:
    """Escalation decision + channel delivery + CSV row + knowledge feed. Idempotent."""
    state, settings = ctx.state, ctx.settings
    if state.log_row:
        return _summary(state)

    if not state.triage:
        state.triage = escalation.apply_guardrails(_fallback_triage(ctx))
        state.triage["priority_score"] = escalation.compute_priority(state.triage)
    triage = state.triage

    merged = state.kb_merged
    kb_ctx = state.kb_context
    # The Knowledge Base Agent (which read the passages) has the final say on coverage, but it can only
    # claim a match if retrieval actually returned something; without its verdict, retrieval decides.
    retrieved_any = bool(merged["policy_passages"] or merged["resolved_ticket_matches"])
    if kb_ctx and "match_found" in kb_ctx:
        kb_match = bool(kb_ctx["match_found"]) and retrieved_any
    else:
        kb_match = bool(merged["match_found"])

    if not state.resolution or not state.resolution.get("response"):
        state.resolution = {
            "response": (f"Hi {state.ticket.customer_name or 'there'}, thank you for reaching out. "
                         f"We could not prepare an automatic answer for ticket {state.ticket_id}; "
                         "a support specialist will review it and get back to you."),
            "confidence": 0, "used_sources": [], "answer_found": False, "fallback": True,
        }
    resolution = state.resolution

    decision = escalation.decide(triage, {"match_found": kb_match}, resolution, settings,
                                 agent_escalate=agent_escalate, agent_reason=agent_reason,
                                 manager=state.manager_route, grounding=state.grounding)
    state.escalation = decision

    # ---- route to the desk that owns this product line / service (fraud, payment, legal desks first)
    catalog = ctx.store.catalog
    primary = triage.get("product_line") or state.division or "general"
    state.routing = {"product_line": primary, "product_line_name": catalog.name(primary),
                     "product_lines": triage.get("product_lines", [primary]),
                     **catalog.route(primary, triage.get("risk_flags", []))}

    # ---- sources actually used
    valid = {p["source"] for p in merged["policy_passages"]} | {
        p["source"] for p in merged["resolved_ticket_matches"]}
    sources: List[str] = []
    for s in list(kb_ctx.get("sources", [])) + list(resolution.get("used_sources", [])):
        s = s.strip()
        if s in valid and s not in sources:
            sources.append(s)
    if not sources and kb_match:
        sources = [p["source"] for p in merged["policy_passages"][:1]]

    # ---- feed the knowledge base (self-learning RAG)
    feed = {"action": "skipped", "reason": ""}
    policy_sources = [s for s in sources if s.endswith(".md")]
    if not settings.learning_enabled:
        feed["reason"] = "learning disabled"
    elif decision["status"] != "Resolved":
        feed["reason"] = "escalated - waits for human approval"
    elif resolution.get("confidence", 0) < settings.confidence_threshold:
        feed["reason"] = "confidence below threshold"
    elif not policy_sources:
        feed["reason"] = "no policy source to back the answer"
    elif state.grounding and not state.grounding.get("grounded", True):
        feed["reason"] = f"reply not fully grounded ({state.grounding.get('note', '')})"
    else:
        feed = ctx.store.add_learned_solution(
            ticket_id=state.ticket_id, category=triage.get("category", "General"),
            question=f"{triage.get('summary', '')} | {state.ticket.complaint}",
            answer=resolution["response"], sources=", ".join(policy_sources),
            confidence=int(resolution.get("confidence", 0)), approved_by="auto", product_line=primary)
    state.kb_feed = feed

    # ---- deliver on the customer's channel (template mail / WhatsApp / Telegram)
    outgoing = messages.customer_message(state, decision, settings,
                                         line_sla=catalog.sla(primary, triage.get("urgency", "Medium")) or "")
    dispatch_results: List[Dict] = []
    if ctx.dispatcher is not None:
        try:
            dispatch_results = ctx.dispatcher.deliver(state, decision, outgoing)
        except Exception as exc:  # never lose the log row because a channel failed
            dispatch_results = [{"channel": state.ticket.channel, "status": "error", "detail": str(exc)[:200]}]
    state.raw_outputs["customer_message"] = outgoing["text"]
    state.raw_outputs["customer_message_kind"] = outgoing["kind"]
    state.raw_outputs["dispatch"] = dispatch_results  # type: ignore[assignment]

    row = {
        "ticket_id": state.ticket_id,
        "timestamp": state.received_at,
        "channel": state.ticket.channel,
        "customer_name": state.ticket.customer_name or "Not provided",
        "customer_contact": state.ticket.customer_contact,
        "order_id": state.ticket.order_id,
        "complaint": state.ticket.complaint,
        "category": triage.get("category", ""),
        "product_line": primary,
        "product_line_name": catalog.name(primary),
        "product_lines": triage.get("product_lines", [primary]),
        "assigned_team": state.routing["name"],
        "urgency": triage.get("urgency", ""),
        "priority_score": triage.get("priority_score", ""),
        "sentiment": triage.get("sentiment", ""),
        "sentiment_score": triage.get("sentiment_score", ""),
        "language": triage.get("language", ""),
        "summary": triage.get("summary", ""),
        "risk_flags": triage.get("risk_flags", []),
        "similar_tickets": [t["ticket_id"] for t in state.history.get("similar_tickets", [])],
        "kb_sources": sources or "none",
        "kb_match": "yes" if kb_match else "no",
        "citations": "; ".join(f"{c['ref']} {c['source']} > {c['section']}"
                               for c in state.grounding.get("cited", [])),
        "grounding": grounding.describe(state.grounding),
        "parent_ticket": state.ticket.parent_ticket,
        "response": resolution.get("response", ""),
        "confidence": resolution.get("confidence", ""),
        "status": decision["status"],
        "escalation_reason": "; ".join(decision["reasons"]),
        "guardrail": "; ".join(state.guardrail.get("flags", [])),
        "fed_to_kb": "yes" if feed.get("action") in {"added", "updated"} else "no",
        "reply_targets": reply_targets(state.ticket),
        "reply_dispatch": "; ".join(f"{d['channel']}:{d['status']}" for d in dispatch_results) or "ui-only",
        "review_status": "Pending" if decision["status"] == "Escalated" else "N/A",
        "reviewed_by": "",
        "final_reply": outgoing["text"],
        "processing_seconds": state.elapsed,
        "model": ctx.model_name,
        "orchestration": settings.orchestration_mode,
    }
    state.log_row = ctx.ticket_log.append(row)
    state.logged_by = logged_by
    return _summary(state)


def reply_targets(ticket) -> str:
    """Every channel + contact the customer should hear back on (used again when a human approves a reply)."""
    targets = []
    if ticket.channel in {"email", "telegram", "whatsapp"} and ticket.customer_contact:
        targets.append(f"{ticket.channel}:{ticket.customer_contact}")
    for ch in ticket.reply_channels:
        contact = (ticket.reply_contacts or {}).get(ch, "").strip()
        if contact and f"{ch}:{contact}" not in targets:
            targets.append(f"{ch}:{contact}")
    return "; ".join(targets)


def finalize_filtered(ctx: AgentContext, reason: str, label: str, reply: str | None,
                      logged_by: str = "input_guardrail", previous_ticket: str = "") -> str:
    """Record a message the guardrail stopped: status 'Filtered', no agents, no escalation, no learning."""
    state, settings = ctx.state, ctx.settings
    if state.log_row:
        return state.log_row.get("ticket_id", "")
    signal = state.sentiment_signal or sentiment_lib.analyze(state.ticket.complaint)
    state.triage = {"category": "Not a support request", "urgency": "Low", "sentiment": signal.get("sentiment", ""),
                    "sentiment_score": signal.get("sentiment_score", ""), "language": signal.get("language", ""),
                    "summary": f"Filtered: {label}", "risk_flags": [], "product_line": "general",
                    "product_lines": ["general"], "priority_score": 0}
    state.escalation = {"status": "Filtered", "reasons": [], "escalate": False, "agreement": True}
    state.routing = {"product_line": "general", "product_line_name": ctx.store.catalog.name("general"),
                     "name": "-", "reason": "guardrail"}
    state.resolution = {"response": reply or "", "confidence": "", "used_sources": []}
    dispatch_results: List[Dict] = []
    if reply and ctx.dispatcher is not None:
        try:
            for target in [t for t in reply_targets(state.ticket).split("; ") if t]:
                ch, to = target.split(":", 1)
                dispatch_results.append(ctx.dispatcher.send_to_customer(
                    ch, to, state.ticket_id, "Info", reply, state.ticket.complaint, state.ticket.subject,
                    kind="guardrail"))
        except Exception as exc:
            dispatch_results = [{"channel": state.ticket.channel, "status": "error", "detail": str(exc)[:200]}]
    state.raw_outputs["customer_message"] = reply or "(no reply sent)"
    state.raw_outputs["customer_message_kind"] = "guardrail"
    state.raw_outputs["dispatch"] = dispatch_results  # type: ignore[assignment]
    flags = [f"blocked:{reason}"] + list(state.guardrail.get("flags", []))
    row = {
        "ticket_id": state.ticket_id, "timestamp": state.received_at, "channel": state.ticket.channel,
        "customer_name": state.ticket.customer_name or "Not provided",
        "customer_contact": state.ticket.customer_contact,
        "order_id": state.ticket.order_id, "complaint": state.ticket.complaint,
        "category": "Not a support request", "product_line": "general",
        "product_line_name": ctx.store.catalog.name("general"), "product_lines": "general", "assigned_team": "-",
        "urgency": "Low", "priority_score": 0, "sentiment": state.triage["sentiment"],
        "sentiment_score": state.triage["sentiment_score"], "language": state.triage["language"],
        "summary": f"Filtered: {label}" + (f" (see {previous_ticket})" if previous_ticket else ""),
        "risk_flags": "", "similar_tickets": previous_ticket, "kb_sources": "none", "kb_match": "no",
        "response": reply or "(no reply - {})".format(label), "confidence": "", "status": "Filtered",
        "escalation_reason": "", "guardrail": "; ".join(flags), "fed_to_kb": "no",
        "reply_targets": reply_targets(state.ticket),
        "reply_dispatch": "; ".join(f"{d['channel']}:{d['status']}" for d in dispatch_results) or "none",
        "review_status": "N/A", "reviewed_by": "", "final_reply": reply or "", "processing_seconds": state.elapsed,
        "model": ctx.model_name if logged_by != "input_guardrail" else "rules (no LLM call)",
        "orchestration": settings.orchestration_mode,
    }
    state.log_row = ctx.ticket_log.append(row)
    state.logged_by = logged_by
    return state.ticket_id


def _summary(state) -> str:
    d = state.escalation
    return (f"LOGGED | {state.ticket_id} | status={d.get('status')} | category={state.triage.get('category')} | "
            f"product_line={state.routing.get('product_line', '')} | team={state.routing.get('name', '')} | "
            f"urgency={state.triage.get('urgency')} | reasons={'; '.join(d.get('reasons', [])) or 'none'} | "
            f"fed_to_kb={state.log_row.get('fed_to_kb', 'no')} | dispatch={state.log_row.get('reply_dispatch', '')}")


def create_log_agent(model_client: ChatCompletionClient, ctx: AgentContext, division: str = "general") -> AssistantAgent:
    state = ctx.state

    async def log_ticket(escalate: bool, escalation_reason: str = "") -> str:
        """Record the ticket in the CSV log, apply the escalation rules, deliver the reply on the customer's channel and feed verified solutions to the knowledge base."""
        t0 = time.time()
        await state.wait_for("resolution_agent", timeout=30)  # make sure the reply has been parsed
        reply = (state.resolution or {}).get("response", "")
        if reply and state.kb_context.get("match_found") and grounding.verifier_enabled(ctx.settings):
            # semantic check of every claim against the retrieved passages, before the escalation decision
            merged = state.kb_merged
            verdict = await grounding.llm_verify(model_client, reply,
                                                 merged["policy_passages"] + merged["resolved_ticket_matches"])
            state.grounding = grounding.merge_llm_verdict(state.grounding, verdict)
        summary = finalize_ticket(ctx, agent_escalate=bool(escalate), agent_reason=escalation_reason)
        state.tool_calls.append({"agent": agent_name(division, NAME), "tool": "log_ticket",
                                 "args": {"escalate": bool(escalate), "escalation_reason": escalation_reason},
                                 "seconds": round(time.time() - t0, 3)})
        return summary

    return AssistantAgent(
        name=agent_name(division, NAME),
        description=f"{DESCRIPTION} ({ctx.store.catalog.name(division)} group)",
        model_client=model_client,
        tools=[log_ticket],
        system_message=system_message(ctx.settings.confidence_threshold) + division_brief(ctx, division),
        reflect_on_tool_use=False,  # the tool's "LOGGED | ..." line is the agent's final message -> clean termination
    )
