"""Sidebar: engine settings, channel status and the System status panel."""
from __future__ import annotations

import os
from typing import Tuple

import streamlit as st

from channels.dispatcher import Dispatcher
from engine.config import DEFAULT_MODELS, Settings, get_settings
from rag.multilingual import LANGUAGE_NAMES

PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                 "groq": "GROQ_API_KEY", "ollama": None, "offline": None}
CHANNEL_NAMES = {"email": "Email", "telegram": "Telegram", "whatsapp": "WhatsApp"}


def render_settings() -> Tuple[Settings, Dispatcher]:
    base = get_settings()
    with st.sidebar:
        st.markdown("### ⚙️ Engine settings")
        providers = list(PROVIDER_KEYS)
        provider = st.selectbox("LLM provider", providers,
                                index=providers.index(base.llm_provider) if base.llm_provider in providers else 0,
                                help="Set LLM_PROVIDER in .env for the default. 'offline' runs the rule-based demo "
                                     "model.")
        default_model = base.llm_model if provider == base.llm_provider and base.llm_model else DEFAULT_MODELS[provider]
        model = st.text_input("Model", value=default_model, disabled=provider == "offline")
        key_var = PROVIDER_KEYS[provider]
        if key_var:
            st.caption(("🔑 " + key_var + " found") if os.getenv(key_var) else ("❌ " + key_var + " missing in .env"))
        orchestration = st.radio("Orchestration", ["round_robin", "selector"],
                                 index=0 if base.orchestration_mode != "selector" else 1, horizontal=True,
                                 help="round_robin = fixed order Query → Knowledge → Resolution → Log. "
                                      "selector = the model picks the next agent (SelectorGroupChat).")
        threshold = st.slider("Escalate below confidence (%)", 50, 95, base.confidence_threshold, 5)
        reply_mode = st.radio("Escalated tickets", ["holding", "send"],
                              index=0 if base.escalated_reply_mode == "holding" else 1, horizontal=True,
                              help="holding = customer gets an acknowledgement, AI draft waits for human approval. "
                                   "send = AI reply is sent with an escalation note.")
        rerankers = ["feature", "cross-encoder", "none"]
        reranker = st.selectbox("Reranker", rerankers,
                                index=rerankers.index(base.reranker) if base.reranker in rerankers else 0,
                                help="Re-orders the fused BM25 + vector candidates. feature = offline, default. "
                                     "cross-encoder = multilingual BAAI/bge-reranker-v2-m3 (pip install "
                                     "sentence-transformers). none = fusion order only.")
        guardrails_on = st.toggle("Guardrails", value=base.guardrails_enabled,
                                  help="Stops greetings, spam, abuse, prompt-injection, gibberish, duplicates and "
                                       "off-topic messages from becoming tickets; masks card numbers / OTPs; cleans "
                                       "AI chatter out of replies.")
        settings = get_settings(llm_provider=provider, llm_model="" if provider == "offline" else model,
                                orchestration_mode=orchestration, confidence_threshold=threshold,
                                escalated_reply_mode=reply_mode, guardrails_enabled=guardrails_on, reranker=reranker)
        st.divider()
        dispatcher = Dispatcher(settings)
        st.markdown("#### Channels")
        st.markdown("\n".join(f"- {'✅' if ok else '🧪'} **{CHANNEL_NAMES.get(name, name.title())}:** "
                              f"`{'live' if ok else 'simulation'}`" for name, ok in dispatcher.status().items()))
    return settings, dispatcher


def render_system_status(app) -> None:
    """What the desk is running on right now (drawn last, so counts include the ticket just processed)."""
    s, store, catalog = app.settings, app.store, app.catalog
    info = store.index_info()
    divisions = catalog.ids()
    log_df = app.ticket_log.read()
    tickets = int((log_df["status"] != "Filtered").sum()) if not log_df.empty else 0
    filtered = int((log_df["status"] == "Filtered").sum()) if not log_df.empty else 0
    embed = {"tfidf": "TF-IDF", "openai": "OpenAI", "sentence-transformers": "multilingual MiniLM"}.get(
        s.embedding_backend, s.embedding_backend)
    langs = ", ".join(LANGUAGE_NAMES[c] for c in ("en", "ta", "hi"))
    lines = [
        f"- **Company:** `{s.company_name}`",
        f"- **LLM:** `{'offline demo' if s.is_offline else s.model_name}`",
        f"- **Agent groups:** `{len(divisions)}` + manager",
        f"- **Agents:** `{1 + 4 * len(divisions)}`",
        f"- **Orchestration:** `{s.orchestration_mode}`",
        f"- **Product lines:** `{sum(l.type == 'product_line' for l in catalog.lines)}`",
        f"- **Services:** `{sum(l.type == 'service' for l in catalog.lines)}`",
        f"- **KB documents:** `{info['documents'] or len(store.documents())}`",
        f"- **KB chunks:** `{info['chunks']}`",
        f"- **Vector DB:** `{info['vector_store']}`",
        f"- **Index:** `{info['origin']}`",
        f"- **Embeddings:** `{embed}`",
        "- **Retriever:** `BM25 + vectors (RRF)`",
        f"- **Reranker:** `{s.reranker}`",
        f"- **Languages:** `{langs} +`",
        f"- **Learned solutions:** `{len(store.learned_entries())}`",
        f"- **Tickets logged:** `{tickets}`",
        f"- **Filtered by guardrail:** `{filtered}`",
        f"- **Guardrails:** `{'on' if s.guardrails_enabled else 'off'}`",
        f"- **Escalate below:** `{s.confidence_threshold}%`",
    ]
    with st.sidebar:
        st.markdown("#### System status")
        st.markdown("\n".join(lines))
