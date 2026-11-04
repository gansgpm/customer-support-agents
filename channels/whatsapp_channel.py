"""WhatsApp channel through the Twilio WhatsApp API (sandbox works for demos).

Inbound, two options:
* polling (default, no public URL needed): list recent messages sent *to* the
  Twilio WhatsApp number through the REST API;
* webhook: run ``uvicorn channels.webhook_server:app`` behind ngrok and set the
  sandbox "WHEN A MESSAGE COMES IN" URL to https://<ngrok>/webhooks/whatsapp.
Outbound: POST to the Messages API.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from channels.base import Channel, InboundMessage

API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def normalize_number(number: str) -> str:
    number = (number or "").strip().replace(" ", "")
    if not number:
        return ""
    return number if number.startswith("whatsapp:") else f"whatsapp:{number if number.startswith('+') else '+' + number}"


def parse_twilio_form(form: Dict) -> Optional[InboundMessage]:
    """Twilio webhook form fields -> InboundMessage."""
    body = (form.get("Body") or "").strip()
    if not body:
        return None
    return InboundMessage(channel="whatsapp", sender_id=form.get("From", ""), sender_name=form.get("ProfileName", ""),
                          text=body, external_id=form.get("MessageSid", ""), meta={"wa_id": form.get("WaId", "")})


class WhatsAppChannel(Channel):
    name = "whatsapp"

    def __init__(self, settings, dry_run: bool = False, session: Optional[requests.Session] = None) -> None:
        super().__init__(settings, dry_run)
        self.session = session or requests.Session()
        self.seen_file = Path(settings.data_dir) / "whatsapp_seen.txt"

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.twilio_account_sid and s.twilio_auth_token and s.twilio_whatsapp_from)

    @property
    def _auth(self):
        return (self.settings.twilio_account_sid, self.settings.twilio_auth_token)

    def fetch_new(self) -> List[InboundMessage]:
        if not self.configured:
            return []
        since = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        resp = self.session.get(API.format(sid=self.settings.twilio_account_sid), auth=self._auth, timeout=20,
                                params={"To": normalize_number(self.settings.twilio_whatsapp_from),
                                        "DateSent>": since, "PageSize": 50})
        resp.raise_for_status()
        seen = set(self.seen_file.read_text().split()) if self.seen_file.exists() else set()
        out = []
        for m in resp.json().get("messages", []):
            if m.get("direction") != "inbound" or m["sid"] in seen or not (m.get("body") or "").strip():
                continue
            out.append(InboundMessage(channel="whatsapp", sender_id=m["from"], sender_name="", text=m["body"].strip(),
                                      external_id=m["sid"], meta={"date_sent": m.get("date_sent", "")}))
            seen.add(m["sid"])
        self.seen_file.parent.mkdir(parents=True, exist_ok=True)
        self.seen_file.write_text("\n".join(sorted(seen)))
        return list(reversed(out))  # oldest first

    def _send(self, to: str, text: str, subject: str = "", html: str = "") -> str:
        sids = []
        for i in range(0, len(text), 1500):  # WhatsApp body limit via Twilio is 1600 chars
            resp = self.session.post(API.format(sid=self.settings.twilio_account_sid), auth=self._auth, timeout=20,
                                     data={"From": normalize_number(self.settings.twilio_whatsapp_from),
                                           "To": normalize_number(to), "Body": text[i:i + 1500]})
            data = resp.json() if resp.content else {}
            if resp.status_code >= 300:
                raise RuntimeError(f"Twilio {resp.status_code}: {data.get('message', '')}"[:200])
            sids.append(data.get("sid", ""))
        return "twilio sid " + ",".join(sids)
