"""Escalation Queue tab: human-in-the-loop review of escalated tickets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.style import chips, esc


def render(app) -> None:
    settings, dispatcher, ticket_log, store = app.settings, app.dispatcher, app.ticket_log, app.store
    st.markdown("#### Escalation queue — human in the loop")
    st.caption("Escalated tickets wait here. A support specialist edits/approves the AI draft; the approved reply is "
               "sent on the customer's channel and stored as a human-verified solution in the knowledge base.")
    df = ticket_log.read()
    pending = df[(df["status"] == "Escalated") & (df["review_status"] == "Pending")].copy()
    desks = sorted(d for d in pending["assigned_team"].unique() if d)
    desk_pick = st.multiselect("Specialist desk", desks, placeholder="All desks", key="queue_desks")
    if desk_pick:
        pending = pending[pending["assigned_team"].isin(desk_pick)]
    if pending.empty:
        st.success("No tickets waiting for review.")
    else:
        pending["p"] = pd.to_numeric(pending["priority_score"], errors="coerce").fillna(0)
        for _, row in pending.sort_values("p", ascending=False).iterrows():
            with st.expander(f"⚠ {row['ticket_id']} · priority {row['priority_score']} · {row['urgency']} · "
                             f"{row['assigned_team'] or 'Support'} · {row['product_line_name'] or row['category']} · "
                             f"{row['channel']} · {row['customer_name'] or row['customer_contact']}"):
                st.markdown(f"**Complaint:** {esc(row['complaint'])}")
                st.markdown("**Reasons:** " + chips(row["escalation_reason"].split(";")), unsafe_allow_html=True)
                st.caption(f"KB sources: {row['kb_sources']} · AI confidence {row['confidence']}% · "
                           f"sentiment {row['sentiment']} ({row['sentiment_score']})")
                draft = st.text_area("Reply to send", value=row["response"], key=f"draft_{row['ticket_id']}", height=170)
                reviewer = st.text_input("Reviewed by", value="Support Specialist", key=f"rev_{row['ticket_id']}")
                c1, c2 = st.columns(2)
                if c1.button("✅ Approve & send", key=f"approve_{row['ticket_id']}", type="primary"):
                    sent = dispatcher.send_approved(row.to_dict(), draft)
                    dispatch_note = "; ".join(f"{d['channel']}:{d['status']}" for d in sent) or "ui-only"
                    learn = "no"
                    if settings.learning_enabled and row["kb_sources"] not in {"", "none"}:
                        store.add_learned_solution(ticket_id=row["ticket_id"], category=row["category"],
                                                   question=f"{row['summary']} | {row['complaint']}", answer=draft,
                                                   sources=row["kb_sources"], confidence=100, approved_by="human",
                                                   product_line=row["product_line"] or "general")
                        learn = "yes (human-approved)"
                    ticket_log.update(row["ticket_id"], {"review_status": "Approved", "reviewed_by": reviewer,
                                                         "final_reply": draft, "fed_to_kb": learn,
                                                         "reply_dispatch": f"{row['reply_dispatch']}; approved→{dispatch_note}"})
                    st.success(f"Approved. Delivery: {dispatch_note}. Knowledge base: {learn}.")
                    st.rerun()
                if c2.button("Mark handled offline", key=f"done_{row['ticket_id']}"):
                    ticket_log.update(row["ticket_id"], {"review_status": "Handled", "reviewed_by": reviewer})
                    st.rerun()
