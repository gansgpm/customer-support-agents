"""Architecture tab: how a complaint flows through the desk."""
from __future__ import annotations

import streamlit as st

from engine.llm import describe


def render(app) -> None:
    settings = app.settings
    st.markdown("#### How a complaint flows")
    st.graphviz_chart("""
digraph G {
  rankdir=TB; nodesep=0.3; ranksep=0.45; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9aa5b1", fillcolor="#f8fafc"];
  edge [color="#7b8794", fontname="Helvetica", fontsize=9];
  subgraph cluster_in { label="Input channels"; style=dashed; color="#c0c7d0";
    web [label="Streamlit form"]; email [label="Email (IMAP)"]; tg [label="Telegram bot"]; wa [label="WhatsApp (Twilio)"]; }
  inbox [label="Unified inbox\\n+ channel worker", fillcolor="#eef4ff"];
  mgr [label="Support Manager agent\\nidentify_product_line · picks the division\\nbrief + ROUTE: <division>", fillcolor="#fff1e0", color="#eb6834"];
  cat [shape=note, label="catalog.json\\ndivisions · keywords · guides · desks · risks", fillcolor="#eef8f1"];
  subgraph cluster_groups { label="Division agent groups (SelectorGroupChat participants, one RoundRobin team each)"; color="#2a78d6";
    gen [label="General group", fillcolor="#e8f1fc"]; ele [label="Electronics group", fillcolor="#e8f1fc"];
    app [label="Home Appliances group", fillcolor="#e8f1fc"]; fas [label="Fashion group", fillcolor="#e8f1fc"];
    gro [label="Remya Fresh Grocery group", fillcolor="#e8f1fc"]; nc [label="Remya Care group", fillcolor="#e8f1fc"];
    ns [label="Remya Shield group", fillcolor="#e8f1fc"]; plus [label="Plus Membership group", fillcolor="#e8f1fc"]; }
  subgraph cluster_inside { label="Inside every group"; color="#9aa5b1"; style=dashed;
    q [label="Query agent\\nsentiment · history RAG"]; k [label="Knowledge agent\\nBM25 + vectors → RRF → rerank\\nscoped policy + learned RAG"];
    r [label="Resolution agent\\ncited reply + confidence\\n→ grounding check"]; l [label="Log agent\\nrules · desk · CSV · notify · learn"];
    q -> k -> r -> l; }
  kb [shape=cylinder, label="knowledge_base\\ngeneral + product-line + service guides\\n+ learned solutions", fillcolor="#fff7e6"];
  csv [shape=cylinder, label="logs/tickets.csv", fillcolor="#fff7e6"];
  out [label="Reply on the same channel"]; hitl [label="Escalation queue\\n(specialist desk approval)", fillcolor="#fdecec"];
  web -> mgr; email -> inbox; tg -> inbox; wa -> inbox; inbox -> mgr; cat -> mgr [style=dashed];
  mgr -> gen; mgr -> ele; mgr -> app; mgr -> fas; mgr -> gro; mgr -> nc; mgr -> ns; mgr -> plus;
  ele -> q [style=dotted, label="e.g."]; k -> kb [dir=both]; l -> csv; l -> out; l -> hitl [label="Escalated"];
  hitl -> out; l -> kb [label="learn"];
}""", width="stretch")
    st.markdown("""
**Escalation rule (Log Agent tool, enforced in code + the agent's own judgement):** escalate when urgency is **High**,
the customer is **very angry**, the Knowledge Base Agent found **no matching document**, the case involves a **legal
threat, fraud, payment dispute or safety issue**, the Resolution Agent's **confidence is below the threshold**,
or the reply states a **figure that is not in the retrieved documents** (grounding check).

**Support Manager & division groups:** the desk is a `SelectorGroupChat` whose first participant is the Support
Manager agent and whose other participants are one nested `RoundRobinGroupChat` per division (general, 4 product lines,
3 services). The Manager calls `identify_product_line`, names the owning division (`ROUTE: <division>`), lists any
secondary divisions and briefs the group; the selector then hands the ticket to that group only. Inside the group the
Knowledge Base Agent's search is scoped to the division's guide (+ secondary divisions + general policies), and the Log
Agent routes escalations to the division's specialist desk (fraud → Trust & Safety, payment disputes → Billing, legal
threats → Customer Relations).

**Guardrails:** (1) an input guardrail runs before any LLM call and stops greetings, gibberish, spam, pure abuse,
prompt-injection attempts, duplicates and flooding (status **Filtered**, polite reply, no agents run); it also masks
card numbers, CVV, OTPs and passwords; (2) the Support Manager can answer `ROUTE: none` for off-topic requests, which
ends the run before any group works, while a real complaint it cannot place is routed to the general group with
*needs human* and escalated; (3) a reply guardrail removes AI self-talk, internal notes and markdown debris.

**Retrieval pipeline (Knowledge Base Agent's `search_kb` tool):** the question - plus an English version when the
customer wrote in Tamil, Hindi or another language (the Query Agent's English summary; a support glossary offline) -
is searched with BM25 and with vectors (FAISS); the rankings are merged with reciprocal rank fusion into 12
candidates, a reranker re-orders them (offline feature reranker by default, multilingual cross-encoder optional),
the ticket's own division and category get a specificity boost and the top 4 go to the agent. Each passage records
how it was found (BM25 / vector ranks, rerank score) and gets a citation number S1, S2, ... The index is saved in
`data/index/` with a manifest and only rebuilt when a document, the embedding model or the chunker changes.

**Grounding check:** after the Resolution Agent writes the reply, the engine drops citations that point to passages
that were never retrieved and checks that every figure in the reply appears in the retrieved text. A reply with an
unsupported figure is escalated; only grounded replies are fed back into the knowledge base.

**Conversation memory:** a follow-up ("any update?", an email reply quoting `[TKT-0012]`, or the same customer
writing again within 7 days) is linked to the earlier ticket; the agents see the earlier turns and the Query Agent
rewrites the follow-up into a self-contained question for the search.

**Observability:** `logs/app.log` (rotating) and `logs/traces.jsonl` (one line per ticket: timings, tool calls,
searches, grounding, tokens) - viewable per ticket in the Ticket Log tab.

**Self-learning RAG:** a ticket resolved with confidence ≥ threshold *and* backed by a policy file is stored as a learned
solution; escalated tickets are stored only after a human approves the reply. Near-duplicates update a `reuse_count`
instead of adding noise, and policy documents always win over learned answers.
""")
    st.markdown(f"Current model: **{describe(settings)}** · orchestration **{settings.orchestration_mode}** · "
                f"confidence threshold **{settings.confidence_threshold}%** · escalated reply mode "
                f"**{settings.escalated_reply_mode}**")
