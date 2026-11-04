# Evaluation Report — AI Customer Support Automation

*Model:* **openai / gpt-4o-mini** · *Orchestration:* **round_robin** · *Run:* 2026-11-04 22:17 · *Cases:* 30 · *Confidence threshold:* 80%

> Automated self-assessment produced by `python -m evaluation.evaluate`. Each test case runs through the real
> AutoGen engine in an isolated workspace; channel delivery is dry-run. The official grade is the evaluator's.

## 1. Rubric summary — 92 / 100

| Area | Marks | Estimate | How it is measured |
|---|---:|---:|---|
| Agent design | 20 | **20.0** (100%) | Support Manager + one 4-agent group per division, single-responsibility prompts and tools; JSON contract; routing and triage accuracy |
| Knowledge grounding | 20 | **14.9** (75%) | agent checks, no invented numbers, honest 'not covered' answers, grounded replies, ground-truth hit@3 and abstention on the HELD-OUT questions |
| Workflow & escalation | 20 | **19.0** (95%) | Manager → division group (Query → Knowledge → Resolution → Log), LOGGED termination + safety nets, escalation accuracy, guardrails |
| Logging | 15 | **14.5** (97%) | row per ticket, required columns, status/reason, KB feed rule |
| Streamlit UI | 15 | **15.0** (100%) | feature checklist (see screenshots for the visual evidence) |
| Code & README | 10 | **9.1** (91%) | structure, no secrets, setup steps, screenshots, tests |

## 2. Node-by-node scores

| Node | Score | Checks (passed / applicable) |
|---|---:|---|
| Guardrails | **96%** | real_complaint_not_stopped 26/26; reply_has_no_ai_chatter 29/30; stopped_when_not_a_ticket 4/4; block_reason_correct 3/4; stopped_by_expected_stage 4/4; no_agent_group_ran_when_stopped 4/4; flag_raised 2/2; pii_not_in_log 1/1 |
| Support Manager | **100%** | json_valid 26/26; product_line_tool_used 26/26; route_line_given 26/26; routed_to_correct_division 26/26; reason_and_guidance 26/26; handed_to_that_group 26/26; secondary_divisions_found 2/2; needs_human_flag 2/2 |
| Customer Query Agent | **100%** | json_valid 26/26; category 26/26; urgency 26/26; sentiment 26/26; sentiment_tool_used 26/26; history_rag_used 26/26; summary_ok 26/26; product_line 26/26; repeat_detected 1/1; all_product_lines_found 2/2; language 1/1 |
| Knowledge Base Agent | **99%** | json_valid 26/26; search_tool_used 26/26; retrieval_hit 24/24; source_shown_in_log 24/24; coverage_decision 24/26; context_brief 26/26; facts_verbatim 26/26; scoped_to_product_line 15/15; learned_reuse 1/1 |
| Resolution Agent | **85%** | json_valid 26/26; numbers_grounded 25/26; required_facts 21/23; mentions_ticket 24/26; polite 26/26; concise 26/26; citations_valid 26/26; reply_grounded 20/26; confidence_calibrated 24/26; honest_when_unknown 0/2 |
| Log Agent | **97%** | row_written 30/30; required_columns 30/30; status_correct 27/30; logged_by_log_agent 26/26; kb_feed_correct 26/26; agent_agrees_with_rules 21/26; product_line_logged 26/26; routed_to_correct_desk 19/19; reason_when_escalated 13/13 |
| Orchestration | **100%** | order_manager_query_kb_resolution_log 26/26; one_division_group_ran 26/26; terminated_on_LOGGED 26/26; within_message_cap 26/26; no_pipeline_error 26/26 |
| Channels (Email / Telegram / WhatsApp) | **100%** | whatsapp_webhook_parsed pass; whatsapp_number_normalised pass; telegram_update_parsed pass; email_parsed_and_quote_stripped pass; email_html_template pass; reply_dispatched_to_3_channels pass; outbox_records_messages pass; inbox_deduplicates pass |

### Escalation decision (confusion matrix)

| | Predicted Escalated | Predicted Resolved |
|---|---:|---:|
| **Expected Escalated** | 10 | 0 |
| **Expected Resolved** | 3 | 13 |

Escalation accuracy: **88%**


### Retrieval against hand-labelled ground truth

87 questions (en, hi, ta) in `evaluation/retrieval_set.json`, each labelled with the passage(s) that answer it, plus questions the documents do not cover. The *dev* split was used to tune query expansion, the reranker and the thresholds; the *held-out test* split is never tuned on and is the honest number. Embeddings: `tfidf`, reranker: `feature`, LLM: `openai / gpt-4o-mini`; production path: **+ LLM rewrite + LLM coverage verdict (production)**.

- Held-out split SHA-256 `c7bfadff79a00eae…` — **unchanged** since it was frozen on 2026-11-04
- Held-out questions found in the code or knowledge base: none
- 'Covered' thresholds: min relevance 0.056, min coverage 0.344, strong relevance 0.396 — calibrated (tfidf.json, 2026-11-04 16:24)

**held-out test**

| variant | hit@1 | hit@3 | hit@3 95% CI | hit@5 | MRR@10 | hit@3 Tamil/Hindi | abstention accuracy | abstention 95% CI | false 'covered' on unanswerable | missed answerable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.536 | 0.75 | 0.57-0.89 | 0.821 | 0.655 | 0.5 | 0.742 | 0.58-0.87 | 0 | 8 |
| Vector only | 0.607 | 0.75 | 0.57-0.89 | 0.857 | 0.696 | 0.5 | 0.742 | 0.58-0.87 | 0 | 8 |
| Hybrid (RRF) | 0.571 | 0.75 | 0.57-0.89 | 0.821 | 0.679 | 0.5 | 0.742 | 0.58-0.87 | 0 | 8 |
| Hybrid + feature reranker | 0.571 | 0.786 | 0.61-0.93 | 0.857 | 0.685 | 0.75 | 0.806 | 0.68-0.94 | 0 | 6 |
| Hybrid + reranker, no translation | 0.536 | 0.679 | 0.50-0.86 | 0.75 | 0.619 | 0.0 | 0.677 | 0.52-0.84 | 0 | 10 |
| + LLM query rewrite | 0.786 | 0.857 | 0.71-0.96 | 0.964 | 0.846 | 0.75 | 0.903 | 0.81-1.00 | 0 | 3 |
| + LLM rewrite + LLM coverage verdict (production) | 0.786 | 0.857 | 0.71-0.96 | 0.964 | 0.846 | 0.75 | 0.903 | 0.77-1.00 | 0 | 3 |

**dev (tuning)**

| variant | hit@1 | hit@3 | hit@3 95% CI | hit@5 | MRR@10 | hit@3 Tamil/Hindi | abstention accuracy | abstention 95% CI | false 'covered' on unanswerable | missed answerable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.902 | 0.98 | 0.94-1.00 | 0.98 | 0.938 | 1.0 | 0.929 | 0.86-0.98 | 1 | 3 |
| Vector only | 0.922 | 0.98 | 0.94-1.00 | 0.98 | 0.948 | 1.0 | 0.929 | 0.86-0.98 | 1 | 3 |
| Hybrid (RRF) | 0.902 | 0.98 | 0.94-1.00 | 0.98 | 0.938 | 1.0 | 0.929 | 0.86-0.98 | 1 | 3 |
| Hybrid + feature reranker | 0.902 | 0.98 | 0.94-1.00 | 0.98 | 0.938 | 1.0 | 0.929 | 0.86-0.98 | 1 | 3 |
| Hybrid + reranker, no translation | 0.706 | 0.784 | 0.67-0.90 | 0.784 | 0.742 | 0.0 | 0.75 | 0.62-0.86 | 1 | 13 |
| + LLM query rewrite | 0.824 | 0.98 | 0.94-1.00 | 1.0 | 0.907 | 1.0 | 0.946 | 0.89-1.00 | 2 | 1 |
| + LLM rewrite + LLM coverage verdict (production) | 0.824 | 0.98 | 0.94-1.00 | 1.0 | 0.907 | 1.0 | 0.982 | 0.95-1.00 | 1 | 0 |

hit@k = share of answerable questions whose labelled passage is in the top k; MRR@10 = mean of 1/rank of the first labelled passage; abstention accuracy = covered/not-covered decision correct (answerable must be covered, unanswerable must not); 95% CI = bootstrap interval (2,000 resamples) - with ~30 held-out questions one question moves hit@3 by about 3.5 points, so read the interval, not the third decimal. Per-question results: `evaluation/reports/retrieval_cases.csv`.

### Answer faithfulness

| Measure | Value |
|---|---:|
| Replies checked | 26 |
| Grounded (every figure found in the retrieved passages, valid citations) | 77% |
| Lexical claim support (factual sentences whose key words are in the passages) | 80% |
| Replies with unsupported figures | 1 |
| LLM-as-judge faithfulness (1-5) | 4.92 |
| LLM-as-judge relevance (1-5) | 5 |

Judge: LLM-as-judge (openai / gpt-4o-mini). Per-reply results: `evaluation/reports/faithfulness.csv`.

## 3. Test cases

| Case | Scenario | Channel | Manager → group | Product line → desk | Category · Urgency · Sentiment | KB source | Conf. | Status (expected) | G | M | Q | K | R | L | O |
|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| PDF-1 | Late delivery | web | general_group | general → General Support Desk | Shipping · Medium · Neutral | shipping_policy.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 89% | 100% | 100% |
| PDF-2 | Shoe refund | web | fashion_group | fashion → Fashion Desk | Refund · Low · Neutral | fashion_footwear.md; refund_policy.md | 95 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PDF-3 | Forgot password | web | general_group | general → General Support Desk | Account · Low · Neutral | account_and_login.md | 95 | ✅ Resolved (Resolved) | 50% | 100% | 100% | 100% | 100% | 100% | 100% |
| PDF-4 | Broken blender | web | home_appliances_group | home_appliances → Appliances Desk | Product · Medium · Negative | home_appliances.md; product_faq.md | 95 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PDF-5 | Charged three times + legal threat | web | general_group | general → Billing Desk | Billing · High · Angry | billing_and_payments.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PDF-6 | Corporate gift cards (not in KB) | web | general_group | general → General Support Desk | General · Low · Positive | billing_and_payments.md; product_faq.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 80% | 60% | 89% | 100% |
| CUS-1 | PDF log example: 10 days late, ignored emails | email | general_group | general → General Support Desk | Shipping · High · Angry | shipping_policy.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 89% | 100% | 100% |
| CUS-2 | Torn package, missing item (WhatsApp) | whatsapp | electronics_group | electronics → Electronics Desk | Shipping · Medium · Negative | shipping_policy.md; electronics.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| CUS-3 | Account takeover / fraud (Telegram) | telegram | general_group | general → Trust & Safety Team | Account · High · Negative | account_and_login.md | 100 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| CUS-4 | Promo code not working | web | general_group | general → General Support Desk | Billing · Low · Neutral | billing_and_payments.md | 90 | ❌ Escalated (Resolved) | 100% | 100% | 100% | 100% | 78% | 75% | 100% |
| CUS-5 | Self-learning re-use (similar to PDF-2) | web | fashion_group | fashion → Fashion Desk | Refund · Low · Neutral | fashion_footwear.md; refund_policy.md | 95 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| LIN-1 | Electronics: phone battery swelling (Telegram) | telegram | electronics_group | electronics → Electronics Desk | Product · High · Neutral | electronics.md | 95 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| LIN-2 | Electronics: laptop warranty | web | electronics_group | electronics → Electronics Desk | Product · Medium · Neutral | electronics.md | 90 | ❌ Escalated (Resolved) | 100% | 100% | 100% | 100% | 78% | 78% | 100% |
| LIN-3 | Fashion: sneaker size exchange | web | fashion_group | fashion → Fashion Desk | Refund · Low · Neutral | fashion_footwear.md; refund_policy.md | 95 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| LIN-4 | Grocery: spoiled milk and paneer (WhatsApp) | whatsapp | grocery_group | grocery → Remya Fresh Grocery Desk | Product · Medium · Neutral | remya_fresh_grocery.md; refund_policy.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| LIN-5 | Grocery: food safety (insect found) | web | grocery_group | grocery → Remya Fresh Grocery Desk | Product · High · Negative | remya_fresh_grocery.md; product_faq.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 89% | 100% | 100% |
| SRV-1 | Remya Care: technician missed AC installation | web | installation_repair_group | installation_repair → Remya Care Services Desk | Service · Medium · Neutral | remya_care_installation_repair.md; home_appliances.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| SRV-2 | Remya Shield: how to raise a claim | web | protection_plan_group | protection_plan → Remya Shield Claims Desk | Service · Low · Neutral | remya_shield_protection_plans.md; product_faq.md | 95 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| SRV-3 | Remya Plus: unwanted renewal refund (email) | email | membership_group | membership → Plus Membership Desk | Billing · Medium · Negative | remya_plus_membership.md; refund_policy.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| SRV-4 | Remya Care: technician damaged wall, was rude (WhatsApp) | whatsapp | installation_repair_group | installation_repair → Remya Care Services Desk | Service · Medium · Negative | remya_care_installation_repair.md | 100 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MUL-1 | TV installation pending + wrong shoe size in one order | web | installation_repair_group | installation_repair → Remya Care Services Desk | Product · Medium · Negative | remya_care_installation_repair.md; fashion_footwear.md; refund_policy.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MAN-1 | Typed reason, no division: rude delivery boy -> human | web | general_group | general → General Support Desk | General · Medium · Negative | remya_care_installation_repair.md; shipping_policy.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 80% | 78% | 100% | 100% |
| MAN-2 | Typed reason routed by knowledge base: sour paneer | web | grocery_group | grocery → Remya Fresh Grocery Desk | Product · Medium · Neutral | remya_fresh_grocery.md; shipping_policy.md | 90 | ✅ Resolved (Resolved) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GRD-1 | Greeting only (Telegram) | telegram | (stopped) | general → - | Not a support request · Low · Neutral | none |  | ✅ Filtered (Filtered) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GRD-2 | Spam with a link | web | (stopped) | general → - | Not a support request · Low · Negative | none |  | ✅ Filtered (Filtered) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GRD-3 | Prompt injection only | web | (stopped) | general → - | Not a support request · Low · Negative | none |  | ✅ Filtered (Filtered) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GRD-4 | Off-topic request (stopped by the Support Manager) | web | (stopped) | general → - | Not a support request · Low · Neutral | none |  | ✅ Filtered (Filtered) | 80% | 100% | 100% | 100% | 100% | 100% | 100% |
| GRD-5 | Real issue + injection attempt -> Trust & Safety | web | general_group | general → Trust & Safety Team | Shipping · Medium · Negative | shipping_policy.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 100% | 89% | 100% |
| GRD-6 | Card number + OTP masked, payment dispute (email) | email | general_group | general → Billing Desk | Billing · High · Negative | billing_and_payments.md | 90 | ✅ Escalated (Escalated) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| CUS-6 | Tamil message (language detection, LLM only) | whatsapp | general_group | general → General Support Desk | Shipping · Medium · Neutral | shipping_policy.md | 90 | ❌ Escalated (Resolved) | 100% | 100% | 100% | 100% | 75% | 75% | 100% |

G = Guardrails, M = Support Manager, Q = Customer Query Agent, K = Knowledge Base Agent, R = Resolution Agent, L = Log Agent, O = Orchestration.

### Failed checks

- **PDF-1** · Resolution Agent · `mentions_ticket`
- **PDF-3** · Guardrails · `reply_has_no_ai_chatter`
- **PDF-6** · Knowledge Base Agent · `coverage_decision`
- **PDF-6** · Resolution Agent · `required_facts`
- **PDF-6** · Resolution Agent · `honest_when_unknown`
- **PDF-6** · Resolution Agent · `reply_grounded`
- **PDF-6** · Resolution Agent · `confidence_calibrated`
- **PDF-6** · Log Agent · `agent_agrees_with_rules`
- **CUS-1** · Resolution Agent · `reply_grounded`
- **CUS-4** · Resolution Agent · `numbers_grounded`
- **CUS-4** · Resolution Agent · `reply_grounded`
- **CUS-4** · Log Agent · `status_correct`
- **CUS-4** · Log Agent · `agent_agrees_with_rules`
- **LIN-2** · Resolution Agent · `required_facts`
- **LIN-2** · Resolution Agent · `reply_grounded`
- **LIN-2** · Log Agent · `status_correct`
- **LIN-2** · Log Agent · `agent_agrees_with_rules`
- **LIN-5** · Resolution Agent · `reply_grounded`
- **MAN-1** · Knowledge Base Agent · `coverage_decision`
- **MAN-1** · Resolution Agent · `honest_when_unknown`
- **MAN-1** · Resolution Agent · `confidence_calibrated`
- **GRD-4** · Guardrails · `block_reason_correct`
- **GRD-5** · Log Agent · `agent_agrees_with_rules`
- **CUS-6** · Resolution Agent · `mentions_ticket`
- **CUS-6** · Resolution Agent · `reply_grounded`
- **CUS-6** · Log Agent · `status_correct`
- **CUS-6** · Log Agent · `agent_agrees_with_rules`

## 4. What each node is checked for

| Node | Checks |
|---|---|
| Guardrails | messages that are not tickets (greeting, spam, injection, off-topic, gibberish, duplicate) are stopped by the expected stage with the right reason and no agent group runs; real complaints are never stopped; injection attempts flagged; card numbers / OTPs never reach the log; no AI chatter in replies |
| Support Manager | valid JSON; `identify_product_line` called; `ROUTE:` line given; routed to the correct division's agent group; *needs human* raised when no division or document fits; secondary divisions of a multi-division ticket listed; reason + brief for the group; the selector handed the ticket to exactly that group |
| Customer Query Agent | valid JSON; category / urgency / sentiment match the expected labels; `analyze_sentiment` and `search_ticket_history` (history RAG) called; correct product line / service; every product line of a multi-line ticket found; one-line summary; repeat complaint detected; language detected |
| Knowledge Base Agent | valid JSON; `search_kb` called; expected policy file retrieved (hit@k) and shown in the log; correct covered / not-covered decision; context brief for the Resolution Agent; every key fact is verbatim from a retrieved passage; learned solution re-used; retrieval stayed inside the ticket's product line(s) + general policies |
| Resolution Agent | valid JSON; every number in the reply exists in the retrieved passages or the ticket (no invented policy); required policy facts present; honest answer when the KB has nothing; ticket number; polite; ≤ 220 words; confidence calibrated against the threshold; citations point to retrieved passages; reply passes the runtime grounding check |
| Log Agent | one CSV row per ticket; all required columns filled; status matches the expected result; escalation reason present; recorded through the `log_ticket` tool (not the engine fallback); only resolved, confident, policy-backed answers fed to the KB; agent's own decision agrees with the rule engine; routed to the correct specialist desk; product line + desk logged |
| Orchestration | speaker order Manager → Query → Knowledge → Resolution → Log; exactly one division group ran; stopped by `TextMentionTermination('LOGGED')`; within the message cap; no pipeline error |
| Channels | Twilio WhatsApp webhook parsing + number normalisation; Telegram update parsing; email parsing with quoted-reply stripping; HTML template mail; reply routed to email / WhatsApp / Telegram; outbox audit trail; inbox de-duplication |

## 5. Structural, UI and repository checks

**Agent structure**

- ✅ support_manager + one agent group per division
- ✅ every group = 4 AutoGen AssistantAgents (query, knowledge, resolution, log)
- ✅ support_manager: description + focused system message
- ✅ support_manager: tools ['identify_product_line']
- ✅ query_agent: description + focused system message
- ✅ query_agent: tools ['analyze_sentiment', 'search_ticket_history']
- ✅ knowledge_agent: description + focused system message
- ✅ knowledge_agent: tools ['search_kb']
- ✅ resolution_agent: description + focused system message
- ✅ resolution_agent: tools none
- ✅ log_agent: description + focused system message
- ✅ log_agent: tools ['log_ticket']
- ✅ every group agent knows its division
- ✅ desk stops on LOGGED, with source-match + message-cap safety nets
- ✅ Support Manager guardrail can stop the desk (ROUTE: none)

**Channels**

- ✅ whatsapp_webhook_parsed
- ✅ whatsapp_number_normalised
- ✅ telegram_update_parsed
- ✅ email_parsed_and_quote_stripped
- ✅ email_html_template
- ✅ reply_dispatched_to_3_channels
- ✅ outbox_records_messages
- ✅ inbox_deduplicates

**Streamlit UI**

- ✅ complaint text area + name/order id + submit
- ✅ each agent step shown live (st.status timeline)
- ✅ final reply in highlighted box
- ✅ green Resolved / red Escalated badge
- ✅ ticket log table + download (CSV & Excel)
- ✅ dashboard (category, status, urgency, sentiment, channel)
- ✅ channel inbox + outbox (email/telegram/whatsapp)
- ✅ human-in-the-loop escalation queue
- ✅ product line / service selector + catalog tab
- ✅ dashboard by product line and specialist desk
- ✅ citations + grounding shown with every reply
- ✅ retrieval trace (found by / ranks / rerank) and per-ticket trace viewer
- ✅ reply streamed live while it is written
- ✅ follow-up linking field (conversation memory)
- ✅ ground-truth retrieval metrics shown
- ✅ app split into modules (app.py + ui/tabs)

**Code & README**

- ✅ README: setup, run, architecture, screenshots
- ✅ requirements.txt with pinned versions
- ✅ ground-truth retrieval set (>= 20 questions)
- ✅ .env.example present
- ✅ .env ignored by git
- ❌ no API keys in code
- ✅ one-command start (streamlit run app.py)
- ✅ screenshots folder (>= 3 images)
- ✅ tests folder
- ✅ held-out retrieval split unchanged since it was frozen (SHA-256)
- ✅ no held-out question copied into the code or the knowledge base

## 6. Reproduce

```bash
python -m evaluation.evaluate                     # provider from .env
python -m evaluation.evaluate --provider offline  # no API key
python -m evaluation.evaluate --embedding openai --embedding-model text-embedding-3-small
python -m evaluation.retrieval_eval               # retrieval only (ablation, CIs, integrity)
python -m evaluation.retrieval_eval --calibrate   # re-fit the 'covered' thresholds on the dev split
```

Raw results: `evaluation/reports/case_results.csv`, `node_checks.csv`, `eval_tickets.csv` (workspace `evaluation\runs\20261104_220731`).
