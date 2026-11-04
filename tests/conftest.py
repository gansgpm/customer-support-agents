import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import get_settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """Isolated workspace: own log, learned store and data folder; offline model; no live channels."""
    return get_settings(
        llm_provider="offline", data_dir=tmp_path / "data", log_file=tmp_path / "tickets.csv",
        learned_file=tmp_path / "learned.jsonl", email_address="", email_password="", telegram_bot_token="",
        twilio_account_sid="", twilio_auth_token="", support_team_email="", support_telegram_chat_id="",
        orchestration_mode="round_robin", confidence_threshold=80, escalated_reply_mode="holding",
    )
