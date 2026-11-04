"""Routes the customer's reply back to the channel(s) the ticket came from, and alerts human support."""
from __future__ import annotations

from typing import Dict, List, Optional

from channels.base import Channel
from channels.email_channel import EmailChannel, render_email_html
from channels.telegram_channel import TelegramChannel
from channels.whatsapp_channel import WhatsAppChannel
from engine import messages
from engine.config import Settings

CHANNEL_CLASSES = {"email": EmailChannel, "telegram": TelegramChannel, "whatsapp": WhatsAppChannel}


class Dispatcher:
    def __init__(self, settings: Settings, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self._channels: Dict[str, Channel] = {}

    def channel(self, name: str) -> Optional[Channel]:
        if name not in CHANNEL_CLASSES:
            return None
        if name not in self._channels:
            self._channels[name] = CHANNEL_CLASSES[name](self.settings, dry_run=self.dry_run)
        return self._channels[name]

    def status(self) -> Dict[str, bool]:
        return {name: self.channel(name).configured for name in CHANNEL_CLASSES}

    # ------------------------------------------------------------------
    def _targets(self, ticket) -> List[tuple]:
        targets = []
        if ticket.channel in CHANNEL_CLASSES and ticket.customer_contact:
            targets.append((ticket.channel, ticket.customer_contact))
        for ch in ticket.reply_channels:
            contact = ticket.reply_contacts.get(ch, "")
            if ch in CHANNEL_CLASSES and contact and (ch, contact) not in targets:
                targets.append((ch, contact))
        return targets

    def send_to_customer(self, channel: str, to: str, ticket_id: str, status: str, text: str, complaint: str,
                         subject: str = "", kind: str = "reply", reviewed: bool = False) -> Dict:
        ch = self.channel(channel)
        if ch is None:
            return {"channel": channel, "to": to, "status": "skipped", "detail": "unknown channel", "kind": kind}
        if channel == "email":
            subj = f"Re: {subject} [{ticket_id}]" if subject else f"[{ticket_id}] {self.settings.company_name} Support - {status}"
            html = render_email_html(self.settings.company_name, ticket_id, "Approved" if reviewed else status, text,
                                     complaint, reviewed=reviewed)
            return ch.send(to, text, subj, html, kind).to_dict()
        header = f"{self.settings.company_name} Support - Ticket {ticket_id}\n\n"
        return ch.send(to, header + text, kind=kind).to_dict()

    def deliver(self, state, decision: Dict, outgoing: Dict[str, str]) -> List[Dict]:
        """Called by the Log Agent's tool once the ticket is decided."""
        t = state.ticket
        results = [self.send_to_customer(ch, to, state.ticket_id, decision["status"], outgoing["text"], t.complaint,
                                         t.subject, kind=outgoing["kind"]) for ch, to in self._targets(t)]
        if decision["status"] == "Escalated":
            results += self.alert_support(state, decision)
        return results

    def alert_support(self, state, decision: Dict) -> List[Dict]:
        """Alert the desk that owns the ticket; falls back to the shared support inbox / chat."""
        alert = messages.support_alert(state, decision)
        team = state.routing or {}
        email = team.get("email") or self.settings.support_team_email
        chat = team.get("telegram") or self.settings.support_telegram_chat_id
        subject = f"[ESCALATED] {state.ticket_id} - {team.get('name', 'Support')} - {state.triage.get('category', '')}"
        out = []
        if email:
            out.append(self.channel("email").send(email, alert, subject, kind="support_alert").to_dict())
        if chat:
            out.append(self.channel("telegram").send(chat, alert, kind="support_alert").to_dict())
        return out

    def send_approved(self, row: Dict, reply: str) -> List[Dict]:
        """Human-in-the-loop: deliver a specialist-approved reply on every channel the ticket asked for."""
        targets = [t for t in str(row.get("reply_targets", "")).split("; ") if ":" in t]
        if not targets and row.get("channel") in CHANNEL_CLASSES and row.get("customer_contact"):
            targets = [f"{row['channel']}:{row['customer_contact']}"]
        out = []
        for target in targets:
            channel, contact = target.split(":", 1)
            out.append(self.send_to_customer(channel, contact, row["ticket_id"], "Resolved", reply,
                                             row.get("complaint", ""), kind="approved_reply", reviewed=True))
        return out
