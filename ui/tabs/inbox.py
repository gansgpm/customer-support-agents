"""Channel Inbox tab: connect/test channels, simulate messages, process the unified inbox, outbox."""
from __future__ import annotations

import asyncio

import streamlit as st

from channel_worker import poll_channels, process_inbox_item
from channels.base import InboundMessage, Outbox
from channels.email_channel import render_email_html
from channels.telegram_channel import check_chat_id
from engine.llm import ProviderConfigError
from ui.live import LiveTimeline, render_result
from ui.style import esc


def render(app) -> None:
    settings, dispatcher, engine, inbox = app.settings, app.dispatcher, app.engine, app.inbox
    st.markdown("#### Omni-channel inbox")
    st.caption("Messages from Email (IMAP), Telegram (Bot API) and WhatsApp (Twilio) land here. Replies go back on "
               "the same channel. Run `python channel_worker.py` to do this automatically 24x7.")
    a, b, c = st.columns([1, 1, 2])
    if a.button("📡 Fetch new messages", width="stretch"):
        report = poll_channels(settings)
        st.toast("Fetched: " + ", ".join(f"{k}: {v}" for k, v in report.items()))
    process_all = b.button("▶ Process all new", type="primary", width="stretch")

    with st.expander("🔌 Connect & test Email / Telegram", expanded=not any(dispatcher.status().values())):
        st.markdown("Fill these lines in `.env`, then **stop and restart** `streamlit run app.py` "
                    "(settings are read at start-up). Full steps: README → *Connect Email and Telegram*.")
        st.code("EMAIL_ADDRESS=yourdesk@gmail.com\nEMAIL_APP_PASSWORD=abcdefghijklmnop       # 16-letter App Password\n"
                "SUPPORT_TEAM_EMAIL=you@yourmail.com        # escalation alerts\n\n"
                "TELEGRAM_BOT_TOKEN=123456789:AA...\nSUPPORT_TELEGRAM_CHAT_ID=987654321   # escalation alerts",
                language="ini")
        e_col, t_col = st.columns(2, gap="large")
        with e_col:
            em = dispatcher.channel("email")
            st.markdown(f"**📧 Email** — {'✅ ' + settings.email_address if em.configured else '❌ not configured'}")
            if em.configured:
                st.caption("Reads: " + em.reading_mode())
            if em.configured and st.button("Test Gmail login (IMAP + SMTP)", key="t_email"):
                with st.spinner("Logging in…"):
                    res = em.check()
                (st.success if res["imap"] == "ok" and str(res["smtp"]).startswith("ok") else st.error)(
                    f"IMAP: {res['imap']} · SMTP: {res['smtp']} · unread mails in the last "
                    f"{settings.email_lookback_days} days: {res['unread']}")
            to = st.text_input("Send a test email to", value=settings.support_team_email, key="t_email_to")
            if em.configured and to and st.button("Send test email", key="t_email_send"):
                r = em.send(to, "Connection test from your AI support desk - email replies are working.",
                            "[TKT-TEST] Connection test",
                            render_email_html(settings.company_name, "TKT-TEST", "Info",
                                              "Connection test from your AI support desk.\n\n- Email replies work.",
                                              "connection test"), kind="test")
                (st.success if r.status == "sent" else st.error)(f"{r.status}: {r.detail}")
            st.caption(f"Escalation alerts → {settings.support_team_email or 'SUPPORT_TEAM_EMAIL not set'}")
        with t_col:
            tg = dispatcher.channel("telegram")
            st.markdown(f"**✈️ Telegram** — {'✅ bot token set' if tg.configured else '❌ not configured'}")
            if tg.configured and st.button("Check bot + find my chat ID", key="t_tg"):
                try:
                    me = tg.get_me()
                    st.success(f"Token OK — bot @{me.get('username')}")
                    chats = tg.find_chats()
                    st.session_state["tg_chats"] = chats
                    if chats:
                        st.dataframe(chats, hide_index=True, width="stretch")
                        st.caption("Copy your chat_id into SUPPORT_TELEGRAM_CHAT_ID (for escalation alerts) "
                                   "or use it as the customer's Telegram contact on the New Ticket form.")
                    else:
                        st.info(f"No chats yet. Open Telegram, search @{me.get('username')}, press Start, "
                                "send 'hello', then click again. (Make sure you message THIS bot - the token in "
                                ".env belongs to it.)")
                except Exception as exc:
                    st.error(f"Telegram check failed: {exc}")
            found = st.session_state.get("tg_chats") or []
            default_chat = settings.support_telegram_chat_id or (found[0]["chat_id"] if found else "")
            chat = st.text_input("Send a test message to chat ID", value=default_chat, key="t_tg_to",
                                 help="A number like 987654321 (groups start with -100). Not a phone number, "
                                      "not an @username, not the number at the start of the bot token.")
            warning = check_chat_id(chat, settings.telegram_bot_token) if chat else None
            if warning:
                st.warning(warning + " Click 'Check bot + find my chat ID' to get the right number.")
            if tg.configured and chat and st.button("Send test message", key="t_tg_send", disabled=bool(warning)):
                r = tg.send(chat, "Connection test from your AI support desk - Telegram replies are working.",
                            kind="test")
                if r.status == "sent":
                    st.success(f"Sent to {chat}. Put SUPPORT_TELEGRAM_CHAT_ID={chat} in .env for escalation alerts.")
                else:
                    st.error(r.detail.replace("RuntimeError: ", ""))
            st.caption(f"Escalation alerts → chat {settings.support_telegram_chat_id or 'SUPPORT_TELEGRAM_CHAT_ID not set'}")

    with st.expander("🧪 Simulate an inbound message (no credentials needed)"):
        with st.form("sim_form", clear_on_submit=True):
            s1, s2, s3 = st.columns([1, 1.2, 1.2])
            sim_channel = s1.selectbox("Channel", ["whatsapp", "telegram", "email"])
            sim_sender = s2.text_input("Sender (number / chat id / email)", value="+919876543210")
            sim_name = s3.text_input("Sender name", value="Priya")
            sim_subject = st.text_input("Subject (email only)", value="")
            sim_text = st.text_area("Message", value="My order is 10 days late and nobody replies to my emails.")
            if st.form_submit_button("Add to inbox"):
                sender = sim_sender if sim_channel != "whatsapp" else (
                    sim_sender if sim_sender.startswith("whatsapp:") else f"whatsapp:{sim_sender}")
                inbox.add([InboundMessage(channel=sim_channel, sender_id=sender, sender_name=sim_name,
                                          subject=sim_subject, text=sim_text, meta={"simulated": True})])
                st.toast("Message added to the inbox")

    rows = inbox.list()
    new_rows = [r for r in rows if r["status"] == "new"]
    st.markdown(f"**{len(new_rows)} new** · {len(rows)} total")
    run_area = st.container()
    to_process = [r["id"] for r in new_rows] if process_all else []
    for r in reversed(rows[-25:]):
        icon = {"email": "📧", "telegram": "✈️", "whatsapp": "🟢", "web": "🌐"}.get(r["channel"], "💬")
        cols = st.columns([0.6, 1.6, 4, 1.2, 1])
        cols[0].markdown(icon + " " + r["channel"])
        cols[1].markdown(f"**{esc(r['sender_name'] or r['sender_id'])}**<br><span class='small-muted'>{esc(r['received_at'])}</span>",
                         unsafe_allow_html=True)
        cols[2].markdown(esc(r["text"][:220]))
        cols[3].markdown(f"`{r['status']}` {r.get('ticket_id', '')}")
        if r["status"] == "new" and cols[4].button("Process", key=f"proc_{r['id']}"):
            to_process.append(r["id"])
    for msg_id in to_process:
        with run_area:
            st.markdown(f"##### Processing inbox message `{msg_id}`")
            tl = LiveTimeline()
            try:
                result = asyncio.run(process_inbox_item(msg_id, settings, on_event=tl, engine=engine))
            except ProviderConfigError as exc:
                st.error(str(exc))
                break
            if result:
                render_result(result)
                st.session_state["last_result"], st.session_state["last_events"] = result, tl.events

    st.divider()
    st.markdown("#### Outbox — what each channel delivered")
    out = Outbox(settings).read(50)
    if not out:
        st.caption("Nothing sent yet.")
    for o in reversed(out):
        status_icon = {"sent": "✅", "simulated": "🧪", "error": "❌", "skipped": "⏭️"}.get(o["status"], "•")
        with st.expander(f"{status_icon} {o['time']} · {o['channel']} → {o['to']} · {o['kind']} · {o['status']}"):
            st.caption(o.get("detail", ""))
            if o["channel"] == "email" and o["kind"] != "support_alert":
                st.markdown(f"**Subject:** {esc(o.get('subject', ''))}")
                tid = o.get("subject", "").split("[")[-1].rstrip("]") if "[" in o.get("subject", "") else ""
                st.html(render_email_html(settings.company_name, tid or "TKT", "Resolved" if o["kind"] == "resolution"
                                          else "Escalated" if o["kind"] == "holding" else "Approved", o["text"], ""))
            else:
                st.text(o["text"])
