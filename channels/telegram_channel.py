"""Telegram channel via the Bot API (long polling in, sendMessage out).

Create a bot with @BotFather, copy the token into TELEGRAM_BOT_TOKEN and send
your bot a message. No public URL is needed because we poll getUpdates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

from channels.base import Channel, InboundMessage

API = "https://api.telegram.org/bot{token}/{method}"
HOW_TO_FIND = ("Open Telegram, search your bot, press Start and send 'hello'; then in the app click "
               "'Check bot + find my chat ID' (or run: python -m channels.setup_check) and use the chat_id it shows.")


def check_chat_id(chat_id: str, token: str = "") -> Optional[str]:
    """Return a warning when the value cannot be a Telegram chat ID for this bot, else None."""
    v = str(chat_id or "").strip()
    if not v:
        return "No chat ID given."
    if v.startswith("@"):
        return None  # public channel username - allowed by the Bot API
    if v.startswith("+") or " " in v:
        return f"'{v}' looks like a phone number. Telegram bots cannot message phone numbers - use the numeric chat_id. "
    if not v.lstrip("-").isdigit():
        return f"'{v}' is not a chat ID - it must be a number like 987654321 (a group's starts with -100)."
    bot_id = (token or "").split(":")[0]
    if bot_id and v == bot_id:
        return (f"{v} is your BOT's own ID (the number at the start of the token), not your chat ID. ")
    if v == "93372553":
        return ("93372553 is @BotFather's ID (it appears in the web.telegram.org address bar while the BotFather chat "
                "is open). Bots cannot message BotFather - use YOUR chat_id. ")
    if v == "987654321":
        return "987654321 is only the example from the instructions - use your own chat_id. "
    return None


def explain_error(description: str, chat_id: str, token: str = "") -> str:
    """Turn Telegram's terse API errors into a fix the user can act on."""
    d = (description or "").lower()
    hint = check_chat_id(chat_id, token) or ""
    if "chat not found" in d:
        return (f"Telegram doesn't know chat {chat_id} for this bot. {hint}This happens when the ID is wrong or that "
                f"person / group has never pressed Start on THIS bot. {HOW_TO_FIND}")
    if "bot was blocked" in d:
        return f"Chat {chat_id} blocked the bot. Ask them to unblock it and press Start again."
    if "can't initiate conversation" in d or "bot can't send messages to bots" in d:
        return f"The bot cannot start a chat with {chat_id}. {HOW_TO_FIND}"
    if "unauthorized" in d or "not found" == d.strip():
        return "TELEGRAM_BOT_TOKEN is wrong or was revoked - copy the token again from @BotFather and restart the app."
    if "terminated by other getupdates" in d or "conflict" in d:
        return ("Another program is reading this bot's updates (a second channel_worker, or a webhook is set). "
                "Stop the other worker, or delete the webhook: https://api.telegram.org/bot<TOKEN>/deleteWebhook")
    return description


def parse_update(update: Dict) -> Optional[InboundMessage]:
    msg = update.get("message") or update.get("edited_message")
    if not msg or not msg.get("text"):
        return None
    if msg["text"].strip().startswith("/start"):
        return None
    sender = msg.get("from", {})
    name = " ".join(x for x in [sender.get("first_name", ""), sender.get("last_name", "")] if x).strip()
    return InboundMessage(channel="telegram", sender_id=str(msg["chat"]["id"]), sender_name=name or sender.get("username", ""),
                          text=msg["text"].strip(), external_id=str(update.get("update_id", msg.get("message_id"))),
                          meta={"username": sender.get("username", ""), "message_id": msg.get("message_id")})


class TelegramChannel(Channel):
    name = "telegram"

    def __init__(self, settings, dry_run: bool = False, session: Optional[requests.Session] = None) -> None:
        super().__init__(settings, dry_run)
        self.session = session or requests.Session()
        self.offset_file = Path(settings.data_dir) / "telegram_offset.txt"

    @property
    def configured(self) -> bool:
        return bool(self.settings.telegram_bot_token)

    def _url(self, method: str) -> str:
        return API.format(token=self.settings.telegram_bot_token, method=method)

    def fetch_new(self) -> List[InboundMessage]:
        if not self.configured:
            return []
        offset = int(self.offset_file.read_text()) if self.offset_file.exists() else 0
        resp = self.session.get(self._url("getUpdates"), params={"offset": offset, "timeout": 0}, timeout=20)
        resp.raise_for_status()
        updates = resp.json().get("result", [])
        out = []
        for upd in updates:
            offset = max(offset, int(upd["update_id"]) + 1)
            m = parse_update(upd)
            if m:
                out.append(m)
        self.offset_file.parent.mkdir(parents=True, exist_ok=True)
        self.offset_file.write_text(str(offset))
        return out

    def _send(self, to: str, text: str, subject: str = "", html: str = "") -> str:
        ids = []
        for i in range(0, len(text), 4000):  # Telegram limit is 4096 chars per message
            resp = self.session.post(self._url("sendMessage"), json={"chat_id": to, "text": text[i:i + 4000]},
                                     timeout=20)
            data = resp.json() if resp.content else {}
            if not data.get("ok"):
                raise RuntimeError(explain_error(data.get("description", json.dumps(data)[:200]), to,
                                                 self.settings.telegram_bot_token))
            ids.append(str(data["result"]["message_id"]))
        return "telegram message_id " + ",".join(ids)

    # ------------------------------------------------------------- setup checks
    def get_me(self) -> dict:
        """Verify the token: returns the bot's username."""
        resp = self.session.get(self._url("getMe"), timeout=20)
        data = resp.json() if resp.content else {}
        if not data.get("ok"):
            raise RuntimeError(explain_error(data.get("description", f"HTTP {resp.status_code}"), "",
                                             self.settings.telegram_bot_token))
        return data["result"]

    def find_chats(self) -> list:
        """Chats that messaged the bot.

        Reads pending updates WITHOUT consuming them, plus chats already pulled into the inbox (once
        'Fetch new messages' or the channel worker has run, Telegram no longer returns those updates).
        """
        resp = self.session.get(self._url("getUpdates"), params={"timeout": 0}, timeout=20)
        data = resp.json() if resp.content else {}
        if not data.get("ok"):
            raise RuntimeError(explain_error(data.get("description", f"HTTP {resp.status_code}"), "",
                                             self.settings.telegram_bot_token))
        chats = {}
        inbox_file = Path(self.settings.data_dir) / "inbox.jsonl"
        if inbox_file.exists():
            for line in inbox_file.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("channel") == "telegram" and row.get("sender_id"):
                    chats[row["sender_id"]] = {"chat_id": str(row["sender_id"]), "name": row.get("sender_name", ""),
                                               "type": "from inbox", "last_message": row.get("text", "")[:60]}
        for upd in data.get("result", []):
            msg = upd.get("message") or upd.get("edited_message") or upd.get("channel_post") or {}
            chat = msg.get("chat") or {}
            if chat.get("id") is not None:
                name = chat.get("title") or " ".join(x for x in [chat.get("first_name", ""), chat.get("last_name", "")] if x)
                chats[str(chat["id"])] = {"chat_id": str(chat["id"]), "name": name or chat.get("username", ""),
                                     "type": chat.get("type", ""), "last_message": (msg.get("text") or "")[:60]}
        return list(chats.values())
