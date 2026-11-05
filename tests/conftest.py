import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import get_settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """Isolated workspace: own log, learned store and data folder; offline model; no live channels.

    Everything that changes retrieval or decisions is pinned, so the tests give the same result whatever your .env
    says (e.g. EMBEDDING_BACKEND=openai would otherwise call the OpenAI API and change the rankings)."""
    return get_settings(
        llm_provider="offline", llm_model="", data_dir=tmp_path / "data", log_file=tmp_path / "tickets.csv",
        learned_file=tmp_path / "learned.jsonl", email_address="", email_password="", telegram_bot_token="",
        twilio_account_sid="", twilio_auth_token="", support_team_email="", support_telegram_chat_id="",
        orchestration_mode="round_robin", confidence_threshold=80, escalated_reply_mode="holding",
        embedding_backend="tfidf", embedding_model="", reranker="feature", reranker_model="",
        kb_min_relevance=None, kb_min_coverage=None, kb_strong_relevance=None, kb_top_k=4, kb_candidates=12,
        rrf_k=60, learned_min_relevance=0.35, grounding_verifier="auto", coverage_verdict="auto",
        learning_enabled=True, guardrails_enabled=True, email_subject_filter="", email_mailbox_dedicated=False,
        very_angry_score=-0.5, max_messages=10, followup_window_days=7,
    )
