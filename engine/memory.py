"""Conversation memory: link a customer's follow-up message to their earlier ticket.

A support conversation rarely ends after one message: "Any update?", "Still not received", or an email reply
"Re: [TKT-0012] ...". Answered on its own, such a message is meaningless ("any update" retrieves nothing).
Before the agents run, the engine looks for the conversation it belongs to:

1. an explicit ``parent_ticket`` (typed in the New Ticket form),
2. a ticket number written in the message or the email subject ("Re: [TKT-0012] Connection test"),
3. the same customer (email address / Telegram chat / WhatsApp number) within ``FOLLOWUP_WINDOW_DAYS``,
   when the new message reads like a follow-up ("still", "any update", "again", a short message, ...).

The earlier turns (what the customer asked, what we replied, the status) are added to the task as
*conversation history*. The Query Agent rewrites the follow-up into a ``standalone_query`` (one complete
English question) that the Knowledge Base Agent searches with - no extra LLM call is spent on it, and an
ordinary first message is searched exactly as written.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

TICKET_REF = re.compile(r"\bTKT-\d{4,}\b", re.I)
FOLLOWUP_CUES = re.compile(
    r"\b(still|again|any update|update on|what about|same (issue|problem)|follow[- ]?up|as i (said|mentioned)|"
    r"previous(ly)?|earlier|already (told|said|sent|complained)|you said|you told|no reply|no response|"
    r"not yet|haven'?t (got|received|heard)|yet to (get|receive)|re:)\b", re.I)


def _rows(log_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if log_df is None or log_df.empty:
        return pd.DataFrame()
    df = log_df.fillna("")
    if "status" in df.columns:
        df = df[df["status"] != "Filtered"]
    return df


def looks_like_followup(text: str) -> bool:
    words = re.findall(r"\w+", text or "")
    return bool(FOLLOWUP_CUES.search(text or "")) or 0 < len(words) <= 6


def find_parent(complaint: str, subject: str = "", contact: str = "", explicit: str = "",
                log_df: Optional[pd.DataFrame] = None, window_days: int = 7,
                now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """The earlier ticket this message continues, with the reason it was linked - or None."""
    df = _rows(log_df)
    if df.empty or "ticket_id" not in df.columns:
        return None
    by_id = {str(r["ticket_id"]).upper(): r for r in df.to_dict("records")}
    if explicit and explicit.strip().upper() in by_id:
        return {**by_id[explicit.strip().upper()], "_reason": "ticket number given by the agent"}
    for ref in TICKET_REF.findall(f"{subject} {complaint}"):
        if ref.upper() in by_id:
            return {**by_id[ref.upper()], "_reason": f"customer quoted {ref.upper()}"}
    if contact and "customer_contact" in df.columns and looks_like_followup(f"{subject} {complaint}"):
        same = df[df["customer_contact"].astype(str).str.strip().str.lower() == contact.strip().lower()]
        if not same.empty:
            last = same.iloc[-1].to_dict()
            try:
                when = datetime.strptime(str(last.get("timestamp", ""))[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                when = None
            if when is None or (now or datetime.now()) - when <= timedelta(days=window_days):
                return {**last, "_reason": f"same customer within {window_days} days, follow-up wording"}
    return None


def thread_turns(parent: Dict[str, Any], log_df: Optional[pd.DataFrame], max_turns: int = 3) -> List[Dict[str, str]]:
    """Earlier turns of the conversation, oldest first (follows parent_ticket links back)."""
    df = _rows(log_df)
    by_id = {str(r["ticket_id"]): r for r in df.to_dict("records")} if not df.empty else {}
    turns, current, seen = [], parent, set()
    while current and len(turns) < max_turns and current.get("ticket_id") not in seen:
        seen.add(current.get("ticket_id"))
        turns.append({
            "ticket_id": str(current.get("ticket_id", "")),
            "date": str(current.get("timestamp", ""))[:16],
            "customer": str(current.get("complaint", ""))[:400],
            "summary": str(current.get("summary", ""))[:200],
            "our_reply": str(current.get("final_reply") or current.get("response") or "")[:400],
            "status": str(current.get("status", "")),
            "product_line": str(current.get("product_line", "")),
        })
        current = by_id.get(str(current.get("parent_ticket", "")))
    return list(reversed(turns))


def history_block(thread: Dict[str, Any]) -> str:
    """Text added to the agents' task."""
    if not thread or not thread.get("turns"):
        return ""
    lines = [f"Conversation history - this message is a follow-up to {thread['parent_ticket']} "
             f"({thread['reason']}). Earlier turns, oldest first:"]
    for t in thread["turns"]:
        lines.append(f"- {t['ticket_id']} ({t['date']}, {t['status']}): customer said: \"{t['customer']}\"")
        if t["our_reply"]:
            lines.append(f"  we replied: \"{t['our_reply']}\"")
    lines.append("Use this history to understand what the customer refers to; answer the NEW message.")
    return "\n".join(lines)


def standalone_query(message: str, thread: Dict[str, Any]) -> str:
    """Offline rewrite of a follow-up into a self-contained search query (an LLM does this in the triage)."""
    if not thread or not thread.get("turns"):
        return message
    last = thread["turns"][-1]
    previous = last["summary"] or last["customer"]
    previous = re.sub(r"^\w+ issue:\s*", "", previous)
    return f"{previous} - follow-up: {message}".strip()
