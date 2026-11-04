"""Live, step-by-step rendering of the agent pipeline in Streamlit."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List

import streamlit as st

from engine.state import AGENT_ORDER, AGENT_TITLES, MANAGER, TicketResult
from ui.style import AGENT_ICONS, chips, esc, status_badge, text_html


def _agent_label(agent: str, suffix: str = "", division_name: str = "") -> str:
    group = f" · {division_name} group" if division_name and agent != MANAGER else ""
    return f"{AGENT_ICONS.get(agent, '🤖')} {AGENT_TITLES.get(agent, agent)}{group}" + (f" — {suffix}" if suffix else "")


def _output_suffix(agent: str, parsed: Dict[str, Any], seconds: float) -> str:
    if agent == MANAGER:
        if parsed.get("ticket_type") == "not_support":
            return f"not a support request — stopped ({seconds}s)"
        extra = f" (+ {', '.join(parsed.get('secondary_divisions', []))})" if parsed.get("secondary_divisions") else ""
        human = " · ⚠ needs human" if parsed.get("needs_human") else ""
        return f"routed to {parsed.get('division_name', '?')} group{extra}{human} ({seconds}s)"
    if agent == "query_agent":
        extra = [x for x in parsed.get("product_lines", [])[1:]]
        lines = f" · + {', '.join(extra)}" if extra else ""
        return (f"{parsed.get('category', '?')} · {parsed.get('urgency', '?')} · "
                f"{parsed.get('sentiment', '?')}{lines} ({seconds}s)")
    if agent == "knowledge_agent":
        src = ", ".join(parsed.get("sources", [])) or "no matching document"
        return f"{'match' if parsed.get('match_found') else 'NO MATCH'} · {src} ({seconds}s)"
    if agent == "resolution_agent":
        return f"confidence {parsed.get('confidence', '?')}% ({seconds}s)"
    if agent == "log_agent":
        text = parsed.get("summary", "")
        status = "Escalated" if "status=Escalated" in text else "Resolved" if "status=Resolved" in text else "?"
        return f"{status} ({seconds}s)"
    return f"{seconds}s"


def _render_output(container, agent: str, parsed: Dict[str, Any], raw: str) -> None:
    if agent == MANAGER and parsed.get("ticket_type") == "not_support":
        container.markdown(chips(["🛡️ not a support request", f"reason: {parsed.get('guardrail_reason') or 'off_topic'}"]),
                           unsafe_allow_html=True)
        if parsed.get("reason"):
            container.markdown(f"**Why:** {esc(parsed['reason'])}")
    elif agent == MANAGER:
        kb = parsed.get("kb_check") or {}
        container.markdown(chips([f"➜ {parsed.get('division_name', '')} group",
                                  f"confidence: {parsed.get('route_confidence', '-')}",
                                  f"routed by: {parsed.get('routed_by', '-')}"]
                                 + (["⚠ needs a human"] if parsed.get("needs_human") else [])
                                 + [f"+ {d}" for d in parsed.get("secondary_divisions", [])]
                                 + ([f"Product/service: {parsed['product_or_service']}"]
                                    if parsed.get("product_or_service") else [])
                                 + [f"⚑ {r}" for r in parsed.get("line_risks", [])]), unsafe_allow_html=True)
        if parsed.get("reason"):
            container.markdown(f"**Why:** {esc(parsed['reason'])}")
        if parsed.get("guidance"):
            container.markdown(f"**Brief for the group:** {esc(parsed['guidance'])}")
        if kb:
            container.caption(f"Knowledge-base check: {'covered' if kb.get('covered') else 'NOT covered'} · best guide "
                              f"{kb.get('best_guide') or '-'} · suggested division {kb.get('suggested_division', '-')}")
    elif agent == "query_agent":
        container.markdown(
            chips([f"🛍️ {parsed.get('product_line_name', parsed.get('product_line', ''))}"]
                  + [f"+ {x}" for x in parsed.get("product_lines", [])[1:]]
                  + [f"Category: {parsed.get('category')}", f"Urgency: {parsed.get('urgency')}",
                   f"Sentiment: {parsed.get('sentiment')} ({parsed.get('sentiment_score')})",
                   f"Priority: {parsed.get('priority_score')}/100", f"Language: {parsed.get('language')}"]
                  + [f"⚑ {r}" for r in parsed.get("risk_flags", [])]
                  + (["↻ repeat issue"] if parsed.get("repeat_issue") else [])),
            unsafe_allow_html=True)
        container.markdown(f"**Summary:** {esc(parsed.get('summary', ''))}")
        if parsed.get("urgency_reason"):
            container.caption(f"Urgency reason: {parsed['urgency_reason']}")
    elif agent == "knowledge_agent":
        if parsed.get("context_brief"):
            container.markdown(f"**Context for the Resolution Agent:** {esc(parsed['context_brief'])}")
        for f in parsed.get("key_facts", [])[:6]:
            ref = f"<b>[{esc(f['ref'])}]</b> " if f.get("ref") else ""
            container.markdown(f"- {ref}{esc(f.get('fact', ''))} <span class='small-muted'>· {esc(f.get('source', ''))}"
                               f"</span>", unsafe_allow_html=True)
        trace = parsed.get("_retrieval", [])
        if trace:
            with container.expander("How the passages were found (hybrid search → rerank)"):
                st.dataframe([{"ref": p.get("ref", ""), "document": p["source"], "section": p["section"][:70],
                               "found by": p.get("found_by", ""), "BM25 rank": (p.get("ranks") or {}).get("bm25"),
                               "vector rank": (p.get("ranks") or {}).get("vector"),
                               "fusion rank": (p.get("ranks") or {}).get("fusion"),
                               "rerank": p.get("rerank_score"), "similarity": p.get("relevance"),
                               "coverage": p.get("coverage")} for p in trace],
                             hide_index=True, width="stretch")
                queries = parsed.get("_queries", [])
                if len(queries) > 1:
                    st.caption("Searched as: " + " · ".join(f"“{q[:80]}”" for q in queries))
        if parsed.get("similar_resolutions"):
            container.caption("Re-used learned solutions: " + ", ".join(
                str(s.get("ticket_id", "")) for s in parsed["similar_resolutions"]))
        if parsed.get("gaps"):
            container.caption(f"Gaps: {parsed['gaps']}")
    elif agent == "resolution_agent":
        container.markdown(f"<div class='reply-box ok' style='border-left-color:#2a78d6;background:transparent'>"
                           f"{text_html(parsed.get('response', ''))}</div>", unsafe_allow_html=True)
        container.caption(f"Sources used: {', '.join(parsed.get('used_sources', [])) or 'none'}"
                          + (f" · citations: {', '.join(parsed['citations'])}" if parsed.get("citations") else ""))
    elif agent == "log_agent":
        container.code(parsed.get("summary", raw), language=None)
    with container.popover("Raw agent message"):
        st.code(raw if isinstance(raw, str) else json.dumps(raw, indent=2), language="json")


def partial_reply(streamed: str) -> str:
    """The customer reply inside a JSON answer that is still being streamed ('{"response": "Hi Ana, ...')."""
    m = re.search(r'"response"\s*:\s*"', streamed)
    if not m:
        return ""
    body = streamed[m.end():]
    end = re.search(r'(?<!\\)"', body)
    body = body[:end.start()] if end else body
    return body.replace("\\n", "\n").replace('\\"', '"').rstrip("\\")


class LiveTimeline:
    """Creates one status box per agent and updates it as AutoGen events stream in."""

    def __init__(self, parent=None) -> None:
        parent = parent or st.container()
        self.header = parent.empty()
        self.memory_slot = parent.empty()  # conversation history (follow-ups)
        self.slots = {}
        self.stream_slots: Dict[str, Any] = {}
        self.stream_text: Dict[str, str] = {}
        for a in AGENT_ORDER:
            self.slots[a] = parent.empty()
            if a == MANAGER:
                self.group_slot = parent.empty()
        self.boxes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.division_name = ""
        for a in AGENT_ORDER:
            self.slots[a].markdown(f"<span class='small-muted'>○ {_agent_label(a)} — waiting</span>",
                                   unsafe_allow_html=True)
        self.group_slot.markdown("<span class='small-muted'>↳ division agent group — chosen by the Support "
                                 "Manager</span>", unsafe_allow_html=True)

    def _box(self, agent: str):
        if agent not in self.boxes:
            self.boxes[agent] = self.slots[agent].status(_agent_label(agent, "working...", self.division_name),
                                                         state="running", expanded=True)
        return self.boxes[agent]

    def __call__(self, event: Dict[str, Any]) -> None:
        etype = event["type"]
        if etype not in {"ticket_done", "reply_chunk"}:  # result stored separately; chunks are transient
            self.events.append(dict(event))
        if etype == "ticket_started":
            self.header.caption(f"Ticket **{event['ticket_id']}** · model: {event['model']} · "
                                f"orchestration: {event['orchestration']}")
        elif etype == "agent_started":
            if event.get("division_name") and event["agent"] != MANAGER:
                self.division_name = event["division_name"]
            self._box(event["agent"])
        elif etype == "group_selected":
            self.division_name = event["division_name"]
            self.group_slot.markdown(
                f"<div style='margin:.2rem 0 .5rem 1.2rem;padding:.35rem .7rem;border-left:3px solid #2a78d6;'>"
                f"↳ <b>{esc(event['division_name'])} agent group</b> takes the ticket "
                f"<span class='small-muted'>· team <code>{esc(event['group'])}</code> · desk {esc(event['team'])}</span>"
                f"</div>", unsafe_allow_html=True)
        elif etype == "tool_call":
            args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in (event["args"] or {}).items()) \
                if isinstance(event["args"], dict) else str(event["args"])[:120]
            self._box(event["agent"]).markdown(f"🔧 calls **`{event['tool']}`**({esc(args)})")
        elif etype == "tool_result":
            box = self._box(event["agent"])
            result = event["result"]
            if isinstance(result, (dict, list)):
                box.json(result, expanded=False)
            else:
                box.code(str(result)[:1500], language=None)
        elif etype == "conversation":
            turns = event.get("turns", [])
            lines = "".join(f"<div class='small-muted'>· <b>{esc(t['ticket_id'])}</b> ({esc(t['date'])}, "
                            f"{esc(t['status'])}): {esc(t['customer'][:140])}</div>" for t in turns)
            self.memory_slot.markdown(
                f"<div style='margin:.2rem 0 .6rem 0;padding:.45rem .8rem;border-left:3px solid #8a5cf6;"
                f"background:rgba(138,92,246,.06);border-radius:6px'>↩ <b>Follow-up to {esc(event['parent_ticket'])}"
                f"</b> <span class='small-muted'>· {esc(event['reason'])}</span>{lines}</div>",
                unsafe_allow_html=True)
        elif etype == "reply_chunk":  # tokens streamed by the model while it writes
            agent = event["agent"]
            box = self._box(agent)
            if agent not in self.stream_slots:
                self.stream_slots[agent] = box.empty()
                self.stream_text[agent] = ""
            self.stream_text[agent] += event.get("text", "")
            draft = partial_reply(self.stream_text[agent])
            if draft:
                self.stream_slots[agent].markdown(f"<div class='reply-box' style='white-space:pre-wrap;opacity:.85'>"
                                                  f"{esc(draft[-1500:])}▌</div>", unsafe_allow_html=True)
        elif etype == "agent_output":
            agent = event["agent"]
            box = self._box(agent)
            if agent in self.stream_slots:
                self.stream_slots.pop(agent).empty()
            _render_output(box, agent, event.get("parsed") or {}, event.get("raw", ""))
            box.update(label=_agent_label(agent, _output_suffix(agent, event.get("parsed") or {}, event["seconds"]),
                                          self.division_name),
                       state="complete", expanded=agent == "resolution_agent")
        elif etype == "guardrail_flags":
            self.header.caption("🛡️ Guardrail: " + ", ".join(event["flags"]))
        elif etype == "guardrail_blocked":
            target = self.slots[MANAGER] if event["stage"] == "input guardrail" else self.group_slot
            target.warning(f"🛡️ Stopped by the {event['stage']}: {event['label']}. No ticket work was done "
                           f"by the agent groups.")
            if event["stage"] == "input guardrail":
                for a in AGENT_ORDER[1:]:
                    self.slots[a].empty()
                self.group_slot.empty()
            else:
                for a in AGENT_ORDER[1:]:
                    self.slots[a].empty()
        elif etype == "error":
            st.error(f"Pipeline error: {event['message']} — the ticket was logged and escalated by the engine.")


def replay(events: List[Dict[str, Any]], parent=None) -> None:
    """Re-draw a finished timeline (after Streamlit reruns)."""
    tl = LiveTimeline(parent)
    for e in events:
        tl(e)
    for agent, box in tl.boxes.items():
        if agent not in {e.get("agent") for e in events if e["type"] == "agent_output"}:
            box.update(state="error", label=_agent_label(agent, "did not finish", tl.division_name))
        else:
            box.update(expanded=False)


def run_live(engine, ticket, parent=None) -> tuple[TicketResult, List[Dict[str, Any]]]:
    timeline = LiveTimeline(parent)
    result = asyncio.run(engine.process(ticket, on_event=timeline))
    return result, timeline.events


def render_result(result: TicketResult) -> None:
    s = result.state
    status = result.status
    tone = "ok" if status == "Resolved" else "bad"
    st.markdown(f"### Final outcome &nbsp; {status_badge(status)}", unsafe_allow_html=True)
    if status == "Filtered":
        g = s.guardrail
        stage = "input guardrail (no LLM call)" if s.logged_by == "input_guardrail" else "Support Manager"
        st.markdown(chips([f"Ticket {result.ticket_id}", f"Stopped by: {stage}",
                           f"Reason: {s.log_row.get('summary', '').replace('Filtered: ', '')}"]
                          + [f"🛡️ {f}" for f in g.get("flags", [])]), unsafe_allow_html=True)
        st.markdown("**Reply sent to the customer**" if result.customer_message and
                    result.customer_message != "(no reply sent)" else "**No reply is sent (spam is ignored)**")
        if result.customer_message and result.customer_message != "(no reply sent)":
            st.markdown(f"<div class='reply-box' style='border-left-color:#9a9a95'>{text_html(result.customer_message)}"
                        f"</div>", unsafe_allow_html=True)
        st.caption("Logged with status 'Filtered' — not counted as a ticket, not escalated, not added to the "
                   "knowledge base.")
        return
    if s.guardrail.get("flags"):
        st.markdown(chips([f"🛡️ {f}" for f in s.guardrail["flags"]]), unsafe_allow_html=True)

    st.markdown(chips([f"🛍️ {s.routing.get('product_line_name', '-')}"]
                      + [f"+ {x}" for x in s.triage.get("product_lines", [])[1:]]
                      + [f"👥 {s.routing.get('name', '-')}"]), unsafe_allow_html=True)
    c = st.columns(3)
    c[0].metric("Category", s.triage.get("category", "-"))
    c[1].metric("Urgency", s.triage.get("urgency", "-"))
    c[2].metric("Sentiment", s.triage.get("sentiment", "-"))
    c = st.columns(3)
    c[0].metric("Priority", s.triage.get("priority_score", "-"), help="0-100 priority score")
    c[1].metric("Confidence", f"{s.resolution.get('confidence', '-')}%")
    c[2].metric("Time", f"{s.log_row.get('processing_seconds', s.elapsed)}s")

    st.markdown("**AI reply (Resolution Agent)**")
    st.markdown(f"<div class='reply-box {tone}'>{text_html(result.reply)}</div>", unsafe_allow_html=True)

    if status == "Escalated":
        st.markdown("**Escalation reasons:** " + chips(s.escalation.get("reasons", [])), unsafe_allow_html=True)
        kind = s.raw_outputs.get("customer_message_kind", "")
        if kind == "holding":
            st.info("The customer receives the acknowledgement below now; the AI draft waits in the "
                    "**Escalation Queue** for a specialist to approve or edit.")
            st.markdown(f"<div class='reply-box' style='border-left-color:#6b7280'>"
                        f"{text_html(result.customer_message)}</div>", unsafe_allow_html=True)

    g = s.grounding or {}
    if g:
        badge = "✅ Grounded" if g.get("grounded") else "⚠️ Not grounded"
        st.markdown(chips([badge, f"Claim support {g.get('supported_ratio', 1.0):.0%}"]
                          + ([f"Unsupported figures: {', '.join(g['unsupported_figures'])}"]
                             if g.get("unsupported_figures") else [])
                          + ([f"Removed citations: {', '.join(g['invalid_citations'])}"]
                             if g.get("invalid_citations") else [])
                          + (["🔎 LLM claim check: " + ("passed" if not g.get("unsupported_claims") else
                                                        f"{len(g['unsupported_claims'])} unsupported")]
                             if (g.get("llm_verifier") or {}).get("ran") else [])), unsafe_allow_html=True)
        if g.get("unsupported_claims"):
            st.warning("Claims the documents do not support: " + " · ".join(g["unsupported_claims"]))
        if g.get("cited"):
            with st.expander(f"Citations ({len(g['cited'])}) - the passages this reply is based on"):
                for c in g["cited"]:
                    text = s.citation_refs.get(c["ref"], {}).get("text", "")
                    st.markdown(f"**[{c['ref']}] {c['source']}** › {c['section']}")
                    st.caption(text[:400] + ("…" if len(text) > 400 else ""))
    usage = s.llm_usage or {}
    meta = [f"Ticket {result.ticket_id}", f"KB sources: {s.log_row.get('kb_sources', '-')}",
            f"Fed to knowledge base: {s.log_row.get('fed_to_kb', 'no')}",
            f"Logged by: {s.logged_by}", f"Stop: {s.stop_reason or '-'}"]
    if s.ticket.parent_ticket:
        meta.insert(1, f"Follow-up to {s.ticket.parent_ticket}")
    if usage.get("calls"):
        tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        meta.append(f"LLM calls {usage['calls']}" + (f" · tokens {tokens:,}" if tokens else ""))
    st.markdown(chips(meta), unsafe_allow_html=True)
    if result.dispatch:
        st.markdown("**Channel delivery**")
        st.dataframe([{"channel": d.get("channel"), "to": d.get("to"), "kind": d.get("kind"),
                       "status": d.get("status"), "detail": d.get("detail")} for d in result.dispatch],
                     hide_index=True, width="stretch")
    if result.error:
        st.warning(f"Engine fallback used: {result.error}")
