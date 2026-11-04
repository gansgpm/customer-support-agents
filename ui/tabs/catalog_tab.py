"""Product Lines & Services tab: divisions, desks, risks and the routing detector."""
from __future__ import annotations

import streamlit as st

from ui.style import chips, esc


def render(app) -> None:
    ticket_log, catalog = app.ticket_log, app.catalog
    st.markdown("#### Product lines & services")
    st.caption("Defined in `knowledge_base/catalog.json`. Each division gets its own agent group (chosen by the "
               "Support Manager), policy guide (scoped RAG), detection keywords, specialist desk and line-specific "
               "risks. Add a division there - its agent group is created automatically, no code changes.")
    df = ticket_log.read()
    for group, title in [("product_line", "Product lines"), ("service", "Services")]:
        st.markdown(f"##### {title}")
        entries = [l for l in catalog.lines if l.type == group]
        cols = st.columns(len(entries))
        for col, line in zip(cols, entries):
            t = df[df["product_line"] == line.id]
            with col.container(border=True):
                st.markdown(f"**{'🛠️' if group == 'service' else '🛍️'} {line.name}**")
                st.caption(line.description)
                m1, m2 = st.columns(2)
                m1.metric("Tickets", len(t))
                m2.metric("Escalated", int((t["status"] == "Escalated").sum()))
                st.markdown(f"<span class='small-muted'>Agent group:</span> <code>{esc(line.id)}_group</code><br>"
                            f"<span class='small-muted'>Desk:</span> {esc(line.team_name)}<br>"
                            f"<span class='small-muted'>Guide:</span> {esc(', '.join(line.documents))}",
                            unsafe_allow_html=True)
                risks = [f"{flag}: {', '.join(words[:3])}…" for flag, words in line.risk_keywords.items() if words]
                if risks:
                    st.markdown(chips(["⚑ " + r for r in risks]), unsafe_allow_html=True)
                if line.sla:
                    st.caption("SLA: " + "; ".join(f"{k} {v}" for k, v in line.sla.items()))
    st.markdown("##### Routing rules")
    st.markdown("- Fraud / account takeover → **{}**  \n- Payment disputes → **{}**  \n- Legal threats → **{}**  \n"
                "- Everything else → the desk of the ticket's main product line or service; tickets with no product "
                "line → **{}**".format(catalog.risk_teams['fraud']['name'], catalog.risk_teams['payment_dispute']['name'],
                                       catalog.risk_teams['legal_threat']['name'], catalog.general.team_name))
    st.markdown("##### Try the Support Manager's product-line detector")
    probe = st.text_input("Message", value="My TV installation is pending and the shoes are the wrong size")
    if probe:
        st.json(catalog.identify(probe), expanded=True)
