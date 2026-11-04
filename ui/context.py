"""Objects shared by every tab, built once per Streamlit run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from channels.dispatcher import Dispatcher
from channels.inbox import Inbox
from engine.config import Settings
from engine.llm import ProviderConfigError
from engine.orchestrator import SupportEngine
from engine.state import TicketInput, TicketResult
from rag.knowledge import KnowledgeStore
from ticketing.ticket_log import TicketLog
from ui.live import run_live


@st.cache_resource(show_spinner="Loading the knowledge index…")
def warm_index(kb_dir: str, backend: str, model: str, data_dir: str) -> str:
    """Load (or build and save) the policy index once per process, not on every Streamlit rerun.

    The index itself is cached inside rag.knowledge and on disk (data/index/); this only makes sure the
    first click does not pay for it, and shows a spinner while it loads.
    """
    from engine.config import get_settings

    store = KnowledgeStore(get_settings(kb_dir=Path(kb_dir), embedding_backend=backend, embedding_model=model,
                                        data_dir=Path(data_dir)))
    return store.index_info()["origin"]


@dataclass
class App:
    settings: Settings
    dispatcher: Dispatcher
    engine: SupportEngine = field(init=False)
    ticket_log: TicketLog = field(init=False)
    store: KnowledgeStore = field(init=False)
    inbox: Inbox = field(init=False)
    line_options: Dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        s = self.settings
        warm_index(str(s.kb_dir), s.embedding_backend, s.embedding_model, str(s.data_dir))
        self.engine = SupportEngine(s, dispatcher=self.dispatcher)
        self.ticket_log = TicketLog(s.log_file, s.data_dir)
        self.store = KnowledgeStore(s)
        self.inbox = Inbox(s)
        self.line_options = {"Auto-detect from the message": ""} | {
            f"{'🛠️' if l.type == 'service' else '🛍️'} {l.name}": l.id for l in self.catalog.lines}

    @property
    def catalog(self):
        return self.store.catalog

    def process_with_ui(self, ticket: TicketInput, parent=None) -> Optional[TicketResult]:
        """Run a ticket with the live agent timeline; friendly error instead of a traceback."""
        try:
            result, events = run_live(self.engine, ticket, parent)
        except ProviderConfigError as exc:
            st.error(str(exc))
            return None
        except Exception as exc:  # network down, quota exceeded, ...: tell the user what to do
            st.error(f"The support pipeline could not run: {type(exc).__name__}: {exc}. Check the internet "
                     "connection and the API key in .env, or switch the LLM provider to 'offline' in the sidebar.")
            return None
        st.session_state["last_result"] = result
        st.session_state["last_events"] = events
        return result
