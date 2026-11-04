# Submission kit — AI Customer Support Automation

## 1. Final checks before pushing (Section 9 of the brief)

| Checklist item | Where it is |
|---|---|
| ☑ Four AutoGen agents, each with one job | `agents/query_agent.py`, `knowledge_agent.py`, `resolution_agent.py`, `log_agent.py` — one group of the four per division |
| ☑ Guardrails against unnecessary comments | `engine/guardrail.py` — input guardrail, Support Manager `ROUTE: none`, reply guardrail; status **Filtered** in the log |
| ☑ Typed reasons routed or handed to a human | New Ticket box accepts free text; Support Manager *needs human* → Escalated |
| ☑ Multiple product & service divisions | `agents/manager_agent.py` (Support Manager) + `engine/team.py` (one nested group per division) + `knowledge_base/catalog.json` |
| ☑ Knowledge Agent answers from the documents and shows the file | `search_kb` tool → `kb_sources` column, sources line under every reply |
| ☑ Resolution Agent does not invent policies | system message + evaluator check "every number in the reply exists in the retrieved text" |
| ☑ Every ticket saved with the Section 7 columns | `logs/tickets.csv` (+ CSV / Excel download in the app) |
| ☑ Complicated cases marked Escalated with a reason | `engine/escalation.py` + Log Agent; `escalation_reason` column; red badge |
| ☑ Runs with one command | `streamlit run app.py` |
| ☑ No API keys in GitHub | `.env` is in `.gitignore`; only `.env.example` is committed |
| ☑ Ground-truth retrieval evaluation (RAG Buildathon review) | `evaluation/retrieval_set.json` (87 questions, dev + frozen held-out split with SHA-256 + leakage check, en/ta/hi, unanswerable) → hit@k, MRR, abstention with 95% CIs, ablation; thresholds calibrated on dev only (`evaluation/calibration/`); LLM-as-judge faithfulness |
| ☑ Hybrid search + reranker, multilingual queries | `rag/retriever.py`, `rag/rerank.py`, `rag/multilingual.py` |
| ☑ Citations + post-generation grounding check | `engine/grounding.py`: figure check + LLM claim verifier; ungrounded replies escalate |
| ☑ Conversation memory, persisted index, traces, pinned requirements, modular UI | `engine/memory.py`, `rag/index_store.py`, `engine/observability.py`, `requirements.txt`, `ui/tabs/` |
| ☑ README with setup, run, architecture and screenshots | `README.md` |

Before you push:

1. Put your key in `.env` and run the evaluation with the model you will demo:
   `python -m evaluation.evaluate` → commit the regenerated `evaluation/reports/` (with an LLM this also runs the
   LLM-as-judge faithfulness scoring). Optionally try `EMBEDDING_BACKEND=openai` and re-run
   `python -m evaluation.retrieval_eval` to show the held-out retrieval numbers with neural embeddings.
2. Retake the screenshots with your LLM (the ones in `screenshots/` were taken with the offline demo model) —
   keep the file names so the README table still works. Make sure no key or phone number is visible.
3. Add at least two of your own test cases to the screenshots (CUS-1 … CUS-6 are already in the app's test-case list).

## 2. Push to GitHub

```bash
cd customer-support-agents
git init
git add .
git status            # .env must NOT be listed
git commit -m "AI Customer Support Automation - AutoGen multi-agent support desk"
git branch -M main
git remote add origin https://github.com/<your-user>/customer-support-agents.git
git push -u origin main
```

## 3. WhatsApp group post (attach 3 screenshots: one Resolved, one Escalated, the Ticket Log)

> **AI Customer Support Automation — AutoGen multi-division support desk** ✅
> A Support Manager agent routes every ticket to one of 8 division agent groups (4 product lines, 3 services, general).
> Each group: Query (sentiment + priority + ticket-history RAG) → Knowledge Base (hybrid BM25 + FAISS RAG scoped to the
> division's guide + learned solutions) → Resolution (grounded reply + confidence) → Log (CSV log, escalation rules,
> routing to the division desk, reply on Email / Telegram / WhatsApp, feeds verified answers back to the KB).
> Streamlit UI with live Manager → group timeline, Resolved/Escalated badges, escalation queue per desk, dashboard by
> division and node-by-node evaluation against the rubric.
> GitHub: https://github.com/<your-user>/customer-support-agents

## 4. Skool community post

**Title:** Multi-division AI Customer Support Desk: a Support Manager + 8 AutoGen agent groups, self-learning RAG, Email / Telegram / WhatsApp

**Body:**
I built a support desk for a made-up company with four product lines and three services, using AutoGen, Python and
Streamlit. A Support Manager agent reads each ticket and hands it to the right division's agent group (a team nested
inside a SelectorGroupChat). In the group, a Customer Query Agent triages the complaint (category, Low/Medium/High
urgency, sentiment, priority) with a sentiment tool and a RAG over past tickets; a Knowledge Base Agent retrieves
verbatim facts from that division's guide with a hybrid BM25 + FAISS search; the Resolution Agent writes a grounded
reply with a confidence score; and the Log Agent applies the escalation rules, routes escalations to the division's
desk, writes the CSV log behind the dashboard, replies on the customer's channel and feeds confident, policy-backed
answers back into the knowledge base.

The hardest part was deciding when the knowledge base genuinely *does not* cover a question (e.g. corporate gift
cards) instead of letting the model answer from loosely related text. I added an IDF-weighted coverage score next to
the similarity score, plus a rule engine that escalates when there is no match or confidence is low — and an
evaluation harness that checks every number in each reply against the retrieved passages.

GitHub: https://github.com/<your-user>/customer-support-agents
*(attach: Resolved ticket, Escalated ticket, Ticket Log)*

## 5. Demo script (4 minutes)

1. **New Ticket → PDF-4 Broken blender** — the Support Manager routes it to the Home Appliances group; watch the four
   group agents, the tool calls, the verbatim facts, the green badge and "Fed to knowledge base: yes".
2. **MULTI · TV installation + shoe size** — the Manager picks the Remya Care group and lists Fashion and Electronics as
   secondary divisions; the reply answers both parts from both guides.
3. **LINE · Electronics: battery swelling** — product-safety risk, Electronics Desk, holding message.
4. **PDF-5 Charged three times** — red badge, escalation reasons, routed to the Billing Desk.
5. **PDF-6 Gift cards** — "no matching document", honest reply, escalated.
6. **Channel Inbox** — simulate a WhatsApp message, process it, show the Outbox / template mail.
7. **Escalation Queue** — filter by desk, edit and approve a ticket; it becomes a human-verified learned solution.
8. **Type your own reason** in the test-case box: *"the delivery boy was rude"* → needs human; *"hi"* and
   *"write me a poem"* → stopped by the guardrail (no agent group runs).
9. **LANG · Tamil: broken mixie** — Tamil complaint, English policy found via translation; open *How the passages
   were found* (BM25 / vector / fusion ranks, rerank) and the **Citations** under the reply (✅ Grounded).
10. **Follow-up**: type "Any update?" with *Follow-up to ticket* = an earlier ticket → linked conversation.
11. **Ticket Log → Trace** — timings, searches, grounding and tokens for any ticket.
12. **Product Lines & Services**, **Dashboard** and **Evaluation** tabs — divisions and desks, tickets per division,
   rubric estimate, node scores, retrieval ground truth (held-out hit@k, MRR, Tamil/Hindi, ablation) and faithfulness.
