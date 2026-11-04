"""Ticket Log tab: the CSV log with filters, downloads and the per-ticket trace."""
from __future__ import annotations

import io

import streamlit as st

from engine import observability


def render(app) -> None:
    settings, ticket_log = app.settings, app.ticket_log
    df = ticket_log.read()
    st.markdown(f"#### Ticket log &nbsp;<span class='small-muted'>{settings.log_file.relative_to(settings.base_dir) if settings.log_file.is_relative_to(settings.base_dir) else settings.log_file}</span>",
                unsafe_allow_html=True)
    if df.empty:
        st.info("The log is empty.")
    else:
        f = st.columns(5)
        sel = {}
        for col, field, label in zip(f, ["status", "product_line_name", "category", "urgency", "channel"],
                                     ["Status", "Product line", "Category", "Urgency", "Channel"]):
            sel[field] = col.multiselect(label, sorted(v for v in df[field].unique() if v), placeholder="All")
        view = df.copy()
        for field, values in sel.items():
            if values:
                view = view[view[field].isin(values)]
        show = ["ticket_id", "timestamp", "channel", "customer_name", "order_id", "parent_ticket", "complaint",
                "language", "product_line_name", "category", "urgency", "sentiment", "kb_sources", "citations",
                "grounding", "response", "status", "escalation_reason", "guardrail", "assigned_team", "confidence",
                "fed_to_kb", "reply_targets", "review_status", "processing_seconds", "llm_calls", "llm_tokens"]
        show = [c for c in show if c in view.columns]
        st.dataframe(view[show].iloc[::-1], hide_index=True, width="stretch", height=420,
                     column_config={"complaint": st.column_config.TextColumn(width="medium"),
                                    "response": st.column_config.TextColumn(width="large")})
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇ Download CSV", view.to_csv(index=False).encode("utf-8"), "tickets.csv", "text/csv",
                           width="stretch")
        buf = io.BytesIO()
        view.to_excel(buf, index=False, sheet_name="tickets")
        d2.download_button("⬇ Download Excel", buf.getvalue(), "tickets.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

        st.markdown("##### Trace — why the desk answered the way it did")
        st.caption("Every ticket writes one line to logs/traces.jsonl: timings per agent, tool calls, each "
                   "knowledge-base search (queries, passages, how they were found, ranks, rerank scores), the "
                   "grounding check and token usage. Application errors go to logs/app.log.")
        pick = st.selectbox("Ticket", list(view["ticket_id"].iloc[::-1]), index=None, placeholder="Choose a ticket")
        if pick:
            trace = observability.read_trace(settings.log_file.parent, pick)
            if not trace:
                st.info("No trace for this ticket (it was logged before tracing was added).")
            else:
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Total time", f"{trace.get('seconds_total', 0)}s")
                usage = trace.get("llm_usage") or {}
                t2.metric("LLM calls", usage.get("calls", 0))
                t3.metric("Tokens", f"{int(usage.get('prompt_tokens', 0)) + int(usage.get('completion_tokens', 0)):,}")
                g = trace.get("grounding") or {}
                t4.metric("Grounded", "yes" if g.get("grounded") else ("no" if g else "-"))
                if trace.get("seconds_per_agent"):
                    st.bar_chart(trace["seconds_per_agent"], horizontal=True, height=180)
                for i, search in enumerate(trace.get("searches", []), start=1):
                    st.markdown(f"**Search {i}** · {'covered' if search.get('match_found') else 'not covered'} · "
                                f"{search.get('retriever', '')}")
                    st.caption("Queries: " + " · ".join(f"“{q[:90]}”" for q in search.get("queries", [])))
                    st.dataframe([{"ref": p.get("ref"), "document": p.get("source"),
                                   "section": (p.get("section") or "")[:70], "found by": p.get("found_by"),
                                   "BM25": (p.get("ranks") or {}).get("bm25"),
                                   "vector": (p.get("ranks") or {}).get("vector"),
                                   "fusion": (p.get("ranks") or {}).get("fusion"),
                                   "rerank": p.get("rerank_score"), "final": p.get("rank_score")}
                                  for p in search.get("passages", [])], hide_index=True, width="stretch")
                with st.expander("Full trace (JSON)"):
                    st.json(trace, expanded=False)
