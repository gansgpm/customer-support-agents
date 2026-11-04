"""Shared look & feel for the Streamlit app."""
from __future__ import annotations

import html

CSS = """
<style>
:root {
  --ok: #0ca30c; --ok-bg: rgba(12,163,12,.10);
  --bad: #d03b3b; --bad-bg: rgba(208,59,59,.10);
  --ink-muted: #6b7280; --line: rgba(128,128,128,.25);
}
.block-container { padding-top: 3.2rem; max-width: 1280px; }
.app-title { font-size: 1.65rem; font-weight: 750; margin-bottom: 0; letter-spacing: -.01em; }
.app-sub { color: var(--ink-muted); margin-top: .15rem; margin-bottom: .8rem; font-size: .95rem; }
.badge { display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .7rem; border-radius:999px;
         font-weight:700; font-size:.82rem; letter-spacing:.02em; border:1px solid transparent; }
.badge.ok  { color: var(--ok);  background: var(--ok-bg);  border-color: rgba(12,163,12,.35); }
.badge.bad { color: var(--bad); background: var(--bad-bg); border-color: rgba(208,59,59,.35); }
.badge.neutral { color: inherit; background: rgba(128,128,128,.12); border-color: var(--line); }
.reply-box { border-radius: 10px; padding: 1rem 1.15rem; margin: .4rem 0 .8rem 0;
             line-height: 1.55; border: 1px solid var(--line); border-left-width: 5px; font-size: .97rem; }
.reply-box.ok  { border-left-color: var(--ok);  background: var(--ok-bg); }
.reply-box.bad { border-left-color: var(--bad); background: var(--bad-bg); }
.chip { display:inline-block; padding:.12rem .55rem; margin:.1rem .25rem .1rem 0; border-radius:6px;
        background: rgba(128,128,128,.12); font-size:.8rem; }
.chip.ok  { background: var(--ok-bg);  color: var(--ok); }
.chip.bad { background: var(--bad-bg); color: var(--bad); }
.kv { color: var(--ink-muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.small-muted { color: var(--ink-muted); font-size: .85rem; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""

AGENT_ICONS = {
    "support_manager": "🎯",
    "query_agent": "🧭",
    "knowledge_agent": "📚",
    "resolution_agent": "✍️",
    "log_agent": "🗂️",
}


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def text_html(text) -> str:
    """Escape text and keep its line breaks without letting Markdown re-interpret it."""
    return esc(text).replace("\n", "<br>")


def status_badge(status: str) -> str:
    if status == "Resolved":
        return '<span class="badge ok">✔ RESOLVED</span>'
    if status == "Escalated":
        return '<span class="badge bad">⚠ ESCALATED TO HUMAN</span>'
    if status == "Filtered":
        return '<span class="badge neutral">🛡️ STOPPED BY GUARDRAIL — NOT A TICKET</span>'
    return f'<span class="badge neutral">{esc(status)}</span>'


def chips(items) -> str:
    return "".join(f'<span class="chip">{esc(i)}</span>' for i in items if str(i).strip())
