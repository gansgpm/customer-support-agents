"""Unified inbox: inbound messages from every channel land here before the engine processes them."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from filelock import FileLock

from channels.base import InboundMessage
from engine.config import Settings


class Inbox:
    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.data_dir) / "inbox.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.path) + ".lock")

    def _load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _save(self, rows: List[Dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, messages: List[InboundMessage]) -> int:
        """Add messages, skipping ones already seen (same channel + provider id)."""
        added = 0
        with self._lock:
            rows = self._load()
            seen = {(r["channel"], r["external_id"]) for r in rows if r.get("external_id")}
            for m in messages:
                if m.external_id and (m.channel, m.external_id) in seen:
                    continue
                rows.append(m.to_dict())
                seen.add((m.channel, m.external_id))
                added += 1
            self._save(rows)
        return added

    def list(self, status: Optional[str] = None) -> List[Dict]:
        rows = self._load()
        return [r for r in rows if status is None or r["status"] == status]

    def get(self, msg_id: str) -> Optional[InboundMessage]:
        for r in self._load():
            if r["id"] == msg_id:
                return InboundMessage(**r)
        return None

    def update(self, msg_id: str, **fields) -> None:
        with self._lock:
            rows = self._load()
            for r in rows:
                if r["id"] == msg_id:
                    r.update(fields)
            self._save(rows)

    def claim(self, msg_id: str) -> bool:
        """Atomically move a message from new -> processing (so two workers never answer it twice)."""
        with self._lock:
            rows = self._load()
            for r in rows:
                if r["id"] == msg_id and r["status"] == "new":
                    r["status"] = "processing"
                    self._save(rows)
                    return True
        return False
