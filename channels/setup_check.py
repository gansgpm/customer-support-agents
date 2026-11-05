"""Check the Email and Telegram connections from a terminal (nothing is sent unless you ask).

    python -m channels.setup_check                       # test logins, list Telegram chats
    python -m channels.setup_check --email-to you@x.com  # also send a test email
    python -m channels.setup_check --telegram-to 12345   # also send a test Telegram message

Exit code 0 when every configured channel works.
"""
from __future__ import annotations

import argparse
import sys

from channels.email_channel import EmailChannel, render_email_html
from channels.telegram_channel import TelegramChannel, check_chat_id
from engine.config import get_settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email-to", help="send a test email to this address")
    ap.add_argument("--telegram-to", help="send a test Telegram message to this chat id")
    args = ap.parse_args()
    s = get_settings()
    ok = True

    print("== Email ==")
    email = EmailChannel(s)
    if not email.configured:
        print("  not configured: set EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env")
    else:
        res = email.check()
        print(f"  account : {s.email_address}")
        print(f"  IMAP    : {res['imap']}  (unread mails: {res['unread']})")
        print(f"  SMTP    : {res['smtp']}")
        print(f"  reads   : {res['reads']}")
        ok &= res["imap"] == "ok" and res["smtp"] == "ok"
        if args.email_to:
            html = render_email_html(s.company_name, "TKT-TEST", "Info", "This is a test from your AI support desk.\n\n"
                                     "- Email replies are working.", "connection test")
            r = email.send(args.email_to, "This is a test from your AI support desk.", "[TKT-TEST] Connection test",
                           html, kind="test")
            print(f"  test mail -> {args.email_to}: {r.status} {r.detail}")
            ok &= r.status == "sent"
    print(f"  escalation alerts go to: {s.support_team_email or '(SUPPORT_TEAM_EMAIL not set)'}")

    print("== Telegram ==")
    tg = TelegramChannel(s)
    if not tg.configured:
        print("  not configured: set TELEGRAM_BOT_TOKEN in .env")
    else:
        try:
            me = tg.get_me()
            print(f"  bot     : @{me.get('username')} ({me.get('first_name')}) - token ok")
            chats = tg.find_chats()
            if chats:
                print("  chats that messaged the bot (use one as SUPPORT_TELEGRAM_CHAT_ID):")
                for c in chats:
                    print(f"    chat_id={c['chat_id']:>14}  {c['type']:<10} {c['name']}  \"{c['last_message']}\"")
            else:
                print("  no recent chats - open Telegram, search for the bot, press Start and send 'hello', then rerun")
        except Exception as exc:
            print(f"  failed: {exc}")
            ok = False
        for chat_id in [c for c in [s.support_telegram_chat_id, args.telegram_to] if c]:
            warning = check_chat_id(chat_id, s.telegram_bot_token)
            if warning:
                print(f"  WARNING chat ID {chat_id}: {warning}")
                ok = False
        if args.telegram_to:
            r = tg.send(args.telegram_to, "Connection test from your AI support desk - Telegram replies are working.",
                        kind="test")
            print(f"  test message -> {args.telegram_to}: {r.status} {r.detail}")
            ok &= r.status == "sent"
    print(f"  escalation alerts go to chat: {s.support_telegram_chat_id or '(SUPPORT_TELEGRAM_CHAT_ID not set)'}")
    print("\nAll configured channels OK" if ok else "\nSome checks failed - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
