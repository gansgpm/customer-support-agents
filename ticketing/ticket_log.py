"""CSV ticket log – one row per complaint. Used by the Log Agent, the dashboard and the history RAG."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from filelock import FileLock

# Required columns from the assignment (Section 7) come first, then the extras used by the dashboard.
REQUIRED_COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint", "category", "urgency", "sentiment",
    "kb_sources", "response", "status", "escalation_reason",
]
LOG_COLUMNS = [
    "ticket_id", "timestamp", "channel", "customer_name", "customer_contact", "order_id", "complaint",
    "category", "product_line", "product_line_name", "product_lines", "assigned_team", "urgency", "priority_score", "sentiment", "sentiment_score", "language", "summary",
    "risk_flags", "similar_tickets", "parent_ticket", "kb_sources", "kb_match", "citations", "grounding",
    "response", "confidence", "status",
    "escalation_reason", "guardrail", "fed_to_kb", "reply_targets", "reply_dispatch", "review_status", "reviewed_by",
    "final_reply",
    "processing_seconds", "llm_calls", "llm_tokens", "model", "orchestration",
]


class TicketLog:
    def __init__(self, log_file: Path, data_dir: Path) -> None:
        self.log_file = Path(log_file)
        self.counter_file = Path(data_dir) / "ticket_counter.txt"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.counter_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.log_file) + ".lock")

    # ----------------------------------------------------------------- ids
    def reserve_ticket_id(self) -> str:
        """Atomically allocate the next ticket number (TKT-0001, TKT-0002, ...)."""
        with FileLock(str(self.counter_file) + ".lock"):
            current = 0
            if self.counter_file.exists():
                try:
                    current = int(self.counter_file.read_text().strip() or 0)
                except ValueError:
                    current = 0
            if current == 0 and self.log_file.exists():  # recover from an existing log
                try:
                    ids = pd.read_csv(self.log_file, usecols=["ticket_id"], dtype=str, keep_default_na=False)["ticket_id"]
                    current = max((int(i.split("-")[-1]) for i in ids if i.startswith("TKT-")), default=0)
                except Exception:
                    current = 0
            current += 1
            self.counter_file.write_text(str(current))
        return f"TKT-{current:04d}"

    # ------------------------------------------------------------- writing
    def append(self, row: Dict) -> Dict:
        clean = {col: _clean(row.get(col, "")) for col in LOG_COLUMNS}
        clean["timestamp"] = clean["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            new_file = not self.log_file.exists() or self.log_file.stat().st_size == 0
            if not new_file:
                self._migrate_header_if_needed()
            with self.log_file.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(clean)
        return clean

    def update(self, ticket_id: str, updates: Dict) -> bool:
        with self._lock:
            if not self.log_file.exists():
                return False
            df = pd.read_csv(self.log_file, dtype=str, keep_default_na=False).fillna("")
            mask = df["ticket_id"] == ticket_id
            if not mask.any():
                return False
            for key, value in updates.items():
                if key not in df.columns:
                    df[key] = ""
                df.loc[mask, key] = _clean(value)
            df.to_csv(self.log_file, index=False)
        return True

    def _migrate_header_if_needed(self) -> None:
        with self.log_file.open(encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        if header != LOG_COLUMNS:
            df = pd.read_csv(self.log_file, dtype=str, keep_default_na=False).fillna("")
            for col in LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df[LOG_COLUMNS].to_csv(self.log_file, index=False)

    # ------------------------------------------------------------- reading
    def read(self) -> pd.DataFrame:
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            return pd.DataFrame(columns=LOG_COLUMNS)
        df = pd.read_csv(self.log_file, dtype=str, keep_default_na=False).fillna("")
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[LOG_COLUMNS]

    def get(self, ticket_id: str) -> Optional[Dict]:
        df = self.read()
        rows = df[df["ticket_id"] == ticket_id]
        return rows.iloc[-1].to_dict() if not rows.empty else None

    def missing_required(self, row: Dict) -> List[str]:
        missing = []
        for col in REQUIRED_COLUMNS:
            if col in {"escalation_reason", "order_id"}:
                continue  # legitimately empty for resolved tickets / web queries without an order
            if not str(row.get(col, "")).strip():
                missing.append(col)
        if row.get("status") == "Escalated" and not str(row.get("escalation_reason", "")).strip():
            missing.append("escalation_reason")
        if row.get("status") == "Filtered" and not str(row.get("guardrail", "")).strip():
            missing.append("guardrail")
        return missing


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = "; ".join(str(v) for v in value)
    return str(value).replace("\r", " ").strip()
