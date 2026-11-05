"""Central configuration, loaded from environment variables / .env.

Every tunable knob of the support desk lives here so the Streamlit sidebar,
the channel worker and the evaluation harness all read the same values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# override=True: the .env file always wins. VS Code ("python.terminal.useEnvFile") copies .env into a terminal when
# it opens, so a terminal opened before you edited .env would otherwise keep the OLD values (old password, old name).
load_dotenv(BASE_DIR / ".env", override=True)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_threshold(name: str) -> Optional[float]:
    """A number, or None for 'auto' / unset: then the calibrated value for the embedding backend is used."""
    value = (os.getenv(name) or "").strip().lower()
    if value in {"", "auto"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "on"}


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b",
    "offline": "offline-rules-v1",
}


@dataclass(frozen=True)
class Settings:
    # ---- LLM -----------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))  # rate limits / network blips
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 60.0))  # seconds per LLM call
    # after the reply is written, an LLM checks every claim against the retrieved passages (auto = when a real LLM
    # is configured); unsupported claims send the ticket to a human
    grounding_verifier: str = field(default_factory=lambda: _env("GROUNDING_VERIFIER", "auto").lower())
    # every search_kb result is checked by a narrow LLM call: do these passages really answer the customer?
    # (auto = when a real LLM is configured; engine/coverage.py) - the verdict becomes the search's match_found
    coverage_verdict: str = field(default_factory=lambda: _env("COVERAGE_VERDICT", "auto").lower())
    stream_replies: bool = field(default_factory=lambda: _env("STREAM_REPLIES", "true").lower() in {"1", "true", "yes"})
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    # ---- Orchestration -------------------------------------------------
    orchestration_mode: str = field(default_factory=lambda: _env("ORCHESTRATION_MODE", "round_robin").lower())
    max_messages: int = field(default_factory=lambda: _env_int("MAX_MESSAGES", 10))
    termination_word: str = "LOGGED"

    # ---- RAG -----------------------------------------------------------
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "tfidf").lower())
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", ""))
    # 'covered' thresholds: a number, or auto (default) = calibrated for the embedding backend
    # (evaluation/calibration/<backend>.json written by `python -m evaluation.retrieval_eval --calibrate`)
    kb_min_relevance: Optional[float] = field(default_factory=lambda: _env_threshold("KB_MIN_RELEVANCE"))
    kb_strong_relevance: Optional[float] = field(default_factory=lambda: _env_threshold("KB_STRONG_RELEVANCE"))
    kb_top_k: int = field(default_factory=lambda: _env_int("KB_TOP_K", 4))
    followup_window_days: int = field(default_factory=lambda: _env_int("FOLLOWUP_WINDOW_DAYS", 7))
    kb_candidates: int = field(default_factory=lambda: _env_int("KB_CANDIDATES", 12))  # fused pool given to the reranker
    kb_min_coverage: Optional[float] = field(default_factory=lambda: _env_threshold("KB_MIN_COVERAGE"))
    learned_min_relevance: float = field(default_factory=lambda: _env_float("LEARNED_MIN_RELEVANCE", 0.35))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    reranker: str = field(default_factory=lambda: _env("RERANKER", "feature").lower())  # feature | cross-encoder | none
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", ""))  # default BAAI/bge-reranker-v2-m3

    # ---- Escalation & learning ------------------------------------------
    confidence_threshold: int = field(default_factory=lambda: _env_int("CONFIDENCE_THRESHOLD", 80))
    very_angry_score: float = field(default_factory=lambda: _env_float("VERY_ANGRY_SCORE", -0.5))
    escalated_reply_mode: str = field(default_factory=lambda: _env("ESCALATED_REPLY_MODE", "holding").lower())
    learning_enabled: bool = field(default_factory=lambda: _env_bool("LEARNING_ENABLED", True))
    guardrails_enabled: bool = field(default_factory=lambda: _env_bool("GUARDRAILS_ENABLED", True))

    # ---- Paths -----------------------------------------------------------
    base_dir: Path = BASE_DIR
    kb_dir: Path = field(default_factory=lambda: Path(_env("KB_DIR", str(BASE_DIR / "knowledge_base"))))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BASE_DIR / "data"))))
    log_file: Path = field(default_factory=lambda: Path(_env("LOG_FILE", str(BASE_DIR / "logs" / "tickets.csv"))))
    learned_file: Path = field(
        default_factory=lambda: Path(_env("LEARNED_FILE", str(BASE_DIR / "knowledge_base" / "learned" / "learned_solutions.jsonl")))
    )

    # ---- Channels ------------------------------------------------------
    company_name: str = field(default_factory=lambda: _env("COMPANY_NAME", "Remya Industries Ltd"))
    support_signature: str = field(default_factory=lambda: _env("SUPPORT_SIGNATURE", "Remya Industries Ltd Customer Support"))

    email_address: str = field(default_factory=lambda: _env("EMAIL_ADDRESS"))
    email_password: str = field(default_factory=lambda: _env("EMAIL_APP_PASSWORD").replace(" ", ""))  # Google shows it in 4 groups
    email_imap_host: str = field(default_factory=lambda: _env("EMAIL_IMAP_HOST", "imap.gmail.com"))
    email_smtp_host: str = field(default_factory=lambda: _env("EMAIL_SMTP_HOST", "smtp.gmail.com"))
    email_smtp_port: int = field(default_factory=lambda: _env_int("EMAIL_SMTP_PORT", 465))
    email_force_ipv4: bool = field(default_factory=lambda: _env("EMAIL_FORCE_IPV4", "false").lower() in {"1", "true", "yes"})
    email_subject_filter: str = field(default_factory=lambda: _env("EMAIL_SUBJECT_FILTER", ""))
    # true ONLY for a mailbox used by nothing but the desk: then every unread mail becomes a ticket. Otherwise the desk
    # reads only replies to its own tickets ([TKT-...] in the subject) and mails matching EMAIL_SUBJECT_FILTER.
    email_mailbox_dedicated: bool = field(default_factory=lambda: _env_bool("EMAIL_MAILBOX_DEDICATED", False))
    email_lookback_days: int = field(default_factory=lambda: _env_int("EMAIL_LOOKBACK_DAYS", 2))  # unread mail older than this is ignored

    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))

    twilio_account_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_whatsapp_from: str = field(default_factory=lambda: _env("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886"))

    support_team_email: str = field(default_factory=lambda: _env("SUPPORT_TEAM_EMAIL"))
    support_telegram_chat_id: str = field(default_factory=lambda: _env("SUPPORT_TELEGRAM_CHAT_ID"))

    worker_poll_seconds: int = field(default_factory=lambda: _env_int("WORKER_POLL_SECONDS", 20))

    # ---------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, "gpt-4o-mini")

    @property
    def is_offline(self) -> bool:
        return self.llm_provider == "offline"

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.log_file.parent, self.learned_file.parent):
            Path(path).mkdir(parents=True, exist_ok=True)


def get_settings(**overrides) -> Settings:
    settings = Settings()
    if overrides:
        settings = settings.with_overrides(**overrides)
    settings.ensure_dirs()
    return settings
