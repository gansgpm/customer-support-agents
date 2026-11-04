"""Dashboard tab: KPIs and charts from logs/tickets.csv."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import charts


def render(app) -> None:
    ticket_log, store = app.ticket_log, app.store
    df = ticket_log.read()
    if df.empty:
        st.info("No tickets yet — submit a few complaints first.")
    else:
        filtered_df = df[df["status"] == "Filtered"]
        df = df[df["status"] != "Filtered"]  # guardrail-stopped messages are not tickets
        num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
        total = max(len(df), 1)
        resolved = int((df["status"] == "Resolved").sum())
        escalated = int((df["status"] == "Escalated").sum())
        k = st.columns(7)
        k[0].metric("Tickets", len(df))
        k[1].metric("Resolved by AI", f"{resolved / total:.0%}", help=f"{resolved} tickets")
        k[2].metric("Escalated", escalated, help="sent to human support")
        k[3].metric("Pending review", int(((df["status"] == "Escalated") & (df["review_status"] == "Pending")).sum()))
        k[4].metric("Avg confidence", f"{num('confidence').mean():.0f}%" if len(df) else "-")
        k[5].metric("Avg handling time", f"{num('processing_seconds').mean():.1f}s" if len(df) else "-")
        k[6].metric("Guardrail stops", len(filtered_df), help="greetings, spam, abuse, off-topic, duplicates… "
                    "logged with status Filtered, no agents run")
        st.caption(f"Learned solutions in the knowledge base: {len(store.learned_entries())} · "
                   f"average priority {num('priority_score').mean():.0f}/100 · "
                   f"urgency mix: " + ", ".join(f"{u} {int((df['urgency'] == u).sum())}" for u in ["High", "Medium", "Low"]))

        f1, f2, _ = st.columns([1, 1.4, 1.6])
        chan_filter = f1.multiselect("Channel", sorted(df["channel"].unique()), default=None, placeholder="All channels")
        line_names = sorted(v for v in df["product_line_name"].unique() if v)
        line_filter = f2.multiselect("Product line / service", line_names, placeholder="All product lines")
        view = df[df["channel"].isin(chan_filter)] if chan_filter else df
        view = view[view["product_line_name"].isin(line_filter)] if line_filter else view

        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.by_product_line_status(view), width="stretch")
        c2.plotly_chart(charts.desk_workload(view), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.by_category_status(view), width="stretch")
        c2.plotly_chart(charts.over_time(view), width="stretch")
        c3, c4, c5 = st.columns(3)
        c3.plotly_chart(charts.counts_bar(view, "urgency", "Urgency", charts.URGENCY, ["Low", "Medium", "High"]),
                        width="stretch")
        c4.plotly_chart(charts.counts_bar(view, "sentiment", "Sentiment", charts.SENTIMENT,
                                          ["Positive", "Neutral", "Negative", "Angry"]), width="stretch")
        c5.plotly_chart(charts.counts_bar(view, "channel", "Channel", horizontal=True), width="stretch")
        g1, g2 = st.columns(2)
        if (view["status"] == "Escalated").any():
            g1.plotly_chart(charts.escalation_reasons(view), width="stretch")
        if not filtered_df.empty:
            g2.plotly_chart(charts.guardrail_blocks(filtered_df), width="stretch")
        st.markdown("**Open escalations by priority**")
        open_ = view[(view["status"] == "Escalated") & (view["review_status"] == "Pending")].copy()
        open_["priority_score"] = pd.to_numeric(open_["priority_score"], errors="coerce")
        st.dataframe(open_.sort_values("priority_score", ascending=False)[
                         ["ticket_id", "priority_score", "urgency", "assigned_team", "product_line_name", "category",
                          "sentiment", "channel", "summary", "escalation_reason"]], hide_index=True, width="stretch")
