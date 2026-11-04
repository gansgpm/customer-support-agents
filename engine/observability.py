"""Logging and per-ticket traces.

* ``logs/app.log``      - rotating application log (1 MB x 5 files): pipeline errors, index rebuilds, channel
  problems, LLM retries. Set ``LOG_LEVEL=DEBUG`` in .env for more detail.
* ``logs/traces.jsonl`` - one JSON line per ticket: model, timings per agent, every tool call with its
  duration, every knowledge-base search (queries, passages with how they were found, ranks and rerank scores,
  the covered / not-covered decision), the grounding check, token usage and errors. It answers "why did the
  desk say that?" for any ticket, and the Ticket Log tab shows it.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

_CONFIGURED: Dict[str, bool] = {}


def setup_logging(log_dir: Path, level: str = "INFO") -> Path:
    """Idempotent: Streamlit re-runs the script on every click, so handlers are added only once."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "app.log"
    if _CONFIGURED.get(str(path)):
        return path
    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "autogen_core", "autogen_agentchat",
                  "autogen_core.events", "watchdog", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED[str(path)] = True
    return path


def _passage(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: p.get(k) for k in ("ref", "source", "section", "found_by", "ranks", "rerank_score", "rank_score",
                                  "relevance", "coverage", "chunk_id")}


def build_trace(state, model: str, orchestration: str, error: Optional[str] = None) -> Dict[str, Any]:
    searches: List[Dict[str, Any]] = [{
        "queries": r.get("search_queries", []), "language": r.get("query_language", ""),
        "scope": r.get("scoped_to", []), "retriever": r.get("retriever", ""),
        "match_found": r.get("match_found"), "top_relevance": r.get("top_relevance"),
        "top_coverage": r.get("top_coverage"),
        "passages": [_passage(p) for p in r.get("policy_passages", [])],
        "learned": [_passage(p) for p in r.get("resolved_ticket_matches", [])],
    } for r in state.kb_results]
    g = state.grounding or {}
    return {
        "ticket_id": state.ticket_id,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model, "orchestration": orchestration,
        "status": state.escalation.get("status") or state.log_row.get("status", ""),
        "division": state.division, "parent_ticket": state.ticket.parent_ticket,
        "language": state.triage.get("language", ""),
        "seconds_total": state.elapsed, "seconds_per_agent": state.agent_seconds,
        "tool_calls": [{k: v for k, v in t.items() if k != "args"} | {"args": t.get("args")}
                       for t in state.tool_calls],
        "searches": searches,
        "grounding": {k: g.get(k) for k in ("grounded", "note", "valid_citations", "invalid_citations",
                                            "unsupported_figures", "unsupported_claims", "supported_ratio",
                                            "llm_verifier")} if g else {},
        "escalation_reasons": state.escalation.get("reasons", []),
        "llm_usage": state.llm_usage,
        "speaker_order": state.speaker_order, "stop_reason": state.stop_reason,
        "parse_errors": state.parse_errors, "error": error,
    }


def write_trace(log_dir: Path, trace: Dict[str, Any]) -> None:
    path = Path(log_dir) / "traces.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")


def read_trace(log_dir: Path, ticket_id: str) -> Optional[Dict[str, Any]]:
    path = Path(log_dir) / "traces.jsonl"
    if not path.exists():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if f'"{ticket_id}"' in line:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ticket_id") == ticket_id:
                found = row
    return found
