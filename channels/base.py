"""Shared types for the omni-channel layer (Email, Telegram, WhatsApp, Web)."""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from filelock import FileLock

from engine.config import Settings


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class InboundMessage:
    channel: str  # email | telegram | whatsapp | web
    sender_id: str  # email address, telegram chat id or whatsapp number - where the reply goes
    text: str
    sender_name: str = ""
    subject: str = ""
    external_id: str = ""  # provider message id, used to avoid processing the same message twice
    received_at: str = field(default_factory=now)
    meta: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "new"  # new | processing | done | error
    ticket_id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DispatchResult:
    channel: str
    to: str
    status: str  # sent | simulated | error | skipped
    detail: str = ""
    kind: str = "reply"  # reply | holding | support_alert | approved_reply

    def to_dict(self) -> Dict:
        return asdict(self)


class Channel(ABC):
    name: str = "channel"

    def __init__(self, settings: Settings, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run

    @property
    @abstractmethod
    def configured(self) -> bool:
        """True when credentials are present in .env."""

    @abstractmethod
    def fetch_new(self) -> List[InboundMessage]:
        """Pull new inbound messages from the provider."""

    @abstractmethod
    def _send(self, to: str, text: str, subject: str = "", html: str = "") -> str:
        """Provider-specific send. Returns a provider id / detail string."""

    def send(self, to: str, text: str, subject: str = "", html: str = "", kind: str = "reply") -> DispatchResult:
        if not to:
            return DispatchResult(self.name, to, "skipped", "no recipient", kind)
        if self.dry_run or not self.configured:
            reason = "dry run" if self.dry_run else f"{self.name} not configured - preview only"
            result = DispatchResult(self.name, to, "simulated", reason, kind)
        else:
            try:
                result = DispatchResult(self.name, to, "sent", self._send(to, text, subject, html)[:200], kind)
            except Exception as exc:
                result = DispatchResult(self.name, to, "error", f"{type(exc).__name__}: {exc}"[:1200], kind)
        Outbox(self.settings).record(result, text, subject)
        return result


class Outbox:
    """Every outbound message (sent or simulated) - lets the UI show what each channel would deliver."""

    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.data_dir) / "outbox.jsonl"

    def record(self, result: DispatchResult, text: str, subject: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": now(), **result.to_dict(), "subject": subject, "text": text}
        with FileLock(str(self.path) + ".lock"):
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read(self, limit: Optional[int] = 200) -> List[Dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return rows[-limit:] if limit else rows
