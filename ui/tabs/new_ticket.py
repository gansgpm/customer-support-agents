"""New Ticket tab: full-width complaint form, then the live agent timeline and the final outcome."""
from __future__ import annotations

import streamlit as st

from engine.state import TicketInput
from ui.live import render_result, replay
from ui.samples import SAMPLES


def render(app) -> None:
    st.markdown("### Submit a customer complaint")
    first_line = list(app.line_options)[0]

    def _fill_sample() -> None:
        pick = (st.session_state.get("sample_pick") or "").strip()
        if not pick:
            return
        for label, text, cname, order in SAMPLES:
            if label == pick:
                st.session_state.update({"complaint": text, "cust_name": cname, "order_id": order,
                                         "line_pick": first_line, "parent_pick": ""})
                return
        # a reason typed by hand: the Support Manager works out where it belongs
        st.session_state.update({"complaint": pick, "cust_name": "", "order_id": "", "line_pick": first_line,
                                 "parent_pick": ""})

    st.selectbox("Load a test case — or type your own reason and press Enter",
                 [s[0] for s in SAMPLES], index=None, key="sample_pick", accept_new_options=True,
                 placeholder="Pick a test case, or type a reason such as 'my geyser is leaking'",
                 on_change=_fill_sample,
                 help="Anything you type that is not in the list becomes the complaint. The Support Manager "
                      "routes it to the right division, sends it to a human if no division fits, and the "
                      "guardrail stops it if it is not a support request. Tamil and Hindi work too.")
    with st.form("ticket_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1, 0.8, 1.3, 0.8])
        name = c1.text_input("Customer name (optional)", key="cust_name")
        order_id = c2.text_input("Order ID (optional)", key="order_id")
        line_label = c3.selectbox("Product line / service", list(app.line_options), key="line_pick",
                                  help="Leave on auto-detect: the Support Manager works it out from the message. "
                                       "Choosing one tells the agents what the customer selected.")
        parent = c4.text_input("Follow-up to ticket", key="parent_pick", placeholder="e.g. TKT-0012",
                               help="Optional. Links this message to an earlier ticket so the agents see the "
                                    "conversation. Detected automatically when the customer quotes the ticket "
                                    "number, or writes again from the same email / chat within "
                                    f"{app.settings.followup_window_days} days ('any update?').")
        complaint = st.text_area("Customer complaint", key="complaint", height=120,
                                 placeholder="e.g. My order is 10 days late and nobody replies to my emails.")
        st.caption("Also send the reply to the customer on (optional — leave blank to show it here only). "
                   "The first contact also identifies the customer for conversation history:")
        r1, r2, r3 = st.columns(3)
        reply_email = r1.text_input("📧 Email", key="contact_email", placeholder="customer@example.com")
        reply_tg = r2.text_input("✈️ Telegram chat ID", key="contact_telegram", placeholder="e.g. 5123456789")
        reply_wa = r3.text_input("🟢 WhatsApp number", key="contact_whatsapp", placeholder="+9198XXXXXXXX")
        submitted = st.form_submit_button("🚀 Submit to the agents", type="primary", width="stretch")

    st.divider()
    st.markdown("### Agent timeline")
    timeline_area = st.container()
    outcome_area = st.container()
    if submitted:
        contacts = {ch: v.strip() for ch, v in
                    [("email", reply_email), ("telegram", reply_tg), ("whatsapp", reply_wa)] if v.strip()}
        if not complaint.strip():
            timeline_area.warning("Please enter a complaint.")
        else:
            ticket = TicketInput(complaint=complaint.strip(), customer_name=name.strip(), order_id=order_id.strip(),
                                 channel="web", customer_contact=next(iter(contacts.values()), ""),
                                 reply_channels=list(contacts), reply_contacts=contacts,
                                 product_line_hint=app.line_options[line_label],
                                 parent_ticket=parent.strip().upper())
            with timeline_area:
                result = app.process_with_ui(ticket)
            if result:
                with outcome_area:
                    render_result(result)
    elif "last_result" in st.session_state:
        with timeline_area:
            replay(st.session_state.get("last_events", []))
        with outcome_area:
            render_result(st.session_state["last_result"])
    else:
        timeline_area.info("Submit a complaint to watch the Support Manager and the division's agents work "
                           "step by step.")
