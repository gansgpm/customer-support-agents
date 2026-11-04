"""Remya Industries Ltd Customer Support – Streamlit UI.

Run:  streamlit run app.py

This file only wires the page together; every tab lives in ui/tabs/<tab>.py with a ``render(app)`` function,
shared objects (settings, engine, ticket log, knowledge store, inbox) are built once in ui/context.py.
"""
from __future__ import annotations

import streamlit as st

from engine.observability import setup_logging
from ui.context import App
from ui.sidebar import render_settings, render_system_status
from ui.style import CSS, esc
from ui.tabs import (architecture, catalog_tab, dashboard, escalation, evaluation_tab, inbox, knowledge,
                     new_ticket, ticket_log_tab)

st.set_page_config(page_title="Remya Industries Ltd Customer Support", page_icon="🎧", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

settings, dispatcher = render_settings()
setup_logging(settings.log_file.parent, settings.log_level)
app = App(settings, dispatcher)
catalog = app.catalog

st.markdown(f"<div class='app-title'>🎧 {esc(settings.company_name.upper())} CUSTOMER SUPPORT</div>"
            f"<div class='app-sub'>Support Manager → division agent group (Query → Knowledge RAG → Resolution → Log) · "
            f"{sum(l.type == 'product_line' for l in catalog.lines)} product lines + "
            f"{sum(l.type == 'service' for l in catalog.lines)} services · Web, Email, Telegram and WhatsApp · "
            f"English, Tamil, Hindi</div>", unsafe_allow_html=True)

TABS = [
    ("🎫 New Ticket", new_ticket), ("📥 Channel Inbox", inbox), ("🧑‍💼 Escalation Queue", escalation),
    ("📊 Dashboard", dashboard), ("📒 Ticket Log", ticket_log_tab), ("🛍️ Product Lines & Services", catalog_tab),
    ("📚 Knowledge Base", knowledge), ("🧪 Evaluation", evaluation_tab), ("🏗️ Architecture", architecture),
]
for tab, (_, module) in zip(st.tabs([label for label, _ in TABS]), TABS):
    with tab:
        module.render(app)

# Drawn last so the counts include the ticket processed in this run (sidebar items still stack under Channels).
render_system_status(app)
