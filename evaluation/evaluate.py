"""Node-by-node evaluation harness mapped to the assignment rubric (100 marks).

    python -m evaluation.evaluate                        # provider from .env
    python -m evaluation.evaluate --provider offline     # no API key needed
    python -m evaluation.evaluate --provider groq --model llama-3.3-70b-versatile --orchestration selector

Every test case runs through the real engine in an isolated workspace
(evaluation/runs/<run_id>/) with channel delivery in dry-run mode, so your
production log and knowledge base are untouched. Results are written to
evaluation/reports/ (EVALUATION.md, latest.json, case_results.csv, node_checks.csv).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from engine.config import BASE_DIR, get_settings
from engine.llm import describe
from engine.orchestrator import SupportEngine
from engine.state import AGENT_ORDER, GROUP_ROLES, MANAGER, TicketInput, division_of, role_of
from rag.text import tokenize

REPORT_DIR = BASE_DIR / "evaluation" / "reports"
RUNS_DIR = BASE_DIR / "evaluation" / "runs"
CASES_FILE = BASE_DIR / "evaluation" / "test_cases.json"
HONEST = ["don't have", "do not have", "not able to", "unable to", "no information", "not covered", "couldn't find",
          "could not find", "don't currently", "not available", "no verified", "don't want to guess"]


# ---------------------------------------------------------------- helpers
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:[.,]\d+)*", text or "")


def _ratio(checks: Dict[str, bool | None]) -> float:
    vals = [v for v in checks.values() if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else 1.0


def _fact_in_passages(fact: str, passages_text: str) -> bool:
    if _norm(fact).strip(" .") in _norm(passages_text):
        return True
    toks = tokenize(fact)
    pool = set(tokenize(passages_text))
    return bool(toks) and sum(t in pool for t in toks) / len(toks) >= 0.85


# ---------------------------------------------------------- per-node checks
def check_guardrail(state, exp) -> Dict[str, bool | None]:
    """Input / Manager / reply guardrails: stop what should be stopped, keep every real complaint."""
    expected_block = exp["status"] == "Filtered"
    blocked = state.escalation.get("status") == "Filtered"
    guard = str(state.log_row.get("guardrail", ""))
    reply = state.resolution.get("response", "") or ""
    return {
        "stopped_when_not_a_ticket": blocked if expected_block else None,
        "real_complaint_not_stopped": (not blocked) if not expected_block else None,
        "block_reason_correct": (f"blocked:{exp['guardrail']}" in guard) if exp.get("guardrail") and expected_block
        else None,
        "stopped_by_expected_stage": (state.logged_by == exp["guard_stage"]) if exp.get("guard_stage") else None,
        "no_agent_group_ran_when_stopped": (not any(role_of(c["agent"]) in GROUP_ROLES for c in state.tool_calls))
        if expected_block else None,
        "flag_raised": (exp["flag"] in guard) if exp.get("flag") else None,
        "pii_not_in_log": (not any(p in state.log_row.get("complaint", "") for p in exp["pii"]))
        if exp.get("pii") else None,
        "reply_has_no_ai_chatter": not re.search(r"(as an ai|language model|LOGGED|ROUTE:|```|key_facts)", reply,
                                                 flags=re.I),
    }


def check_manager(state, exp) -> Dict[str, bool | None]:
    r = state.manager_route
    tools = {c["tool"] for c in state.tool_calls if c["agent"] == MANAGER}
    expected_secondary = [l for l in exp.get("product_lines", [])[1:] if l != "general"]
    return {
        "json_valid": MANAGER not in state.parse_errors,
        "product_line_tool_used": "identify_product_line" in tools,
        "route_line_given": bool(r.get("route_line_found")),
        "routed_to_correct_division": (r.get("division") in exp["product_line"]) if "product_line" in exp else None,
        "needs_human_flag": (bool(r.get("needs_human")) == exp["needs_human"]) if "needs_human" in exp else None,
        "secondary_divisions_found": (set(expected_secondary) <= set(r.get("secondary_divisions", [])))
        if expected_secondary else None,
        "reason_and_guidance": bool(r.get("reason")) and bool(r.get("guidance")),
        "handed_to_that_group": bool(r.get("division")) and state.division == r.get("division"),
    }


def check_query(state, exp) -> Dict[str, bool | None]:
    t = state.triage
    tools = {c["tool"] for c in state.tool_calls if role_of(c["agent"]) == "query_agent"}
    return {
        "json_valid": "query_agent" not in state.parse_errors,
        "category": t.get("category") in exp["category"],
        "urgency": t.get("urgency") in exp["urgency"],
        "sentiment": t.get("sentiment") in exp["sentiment"],
        "sentiment_tool_used": "analyze_sentiment" in tools,
        "history_rag_used": "search_ticket_history" in tools,
        "summary_ok": 0 < len(t.get("summary", "").split()) <= 30,
        "repeat_detected": (bool(t.get("repeat_issue")) == exp["repeat_issue"]) if "repeat_issue" in exp else None,
        "language": (t.get("language", "")[:2] == exp["language"]) if exp.get("language") else None,
        "product_line": (t.get("product_line") in exp["product_line"]) if "product_line" in exp else None,
        "all_product_lines_found": (set(exp["product_lines"]) <= set(t.get("product_lines", [])))
        if "product_lines" in exp else None,
    }


def check_knowledge(state, exp) -> Dict[str, bool | None]:
    merged = state.kb_merged
    kb = state.kb_context
    retrieved = {p["source"] for p in merged["policy_passages"]}
    shown = set(str(state.log_row.get("kb_sources", "")).replace(";", ",").replace(" ", "").split(","))
    passages_text = " ".join(p["text"] for p in merged["policy_passages"] + merged["resolved_ticket_matches"])
    facts = [f["fact"] for f in kb.get("key_facts", [])]
    no_match = exp.get("no_match", False)
    kb_says = bool(kb.get("match_found")) if kb else bool(merged["match_found"])
    return {
        "json_valid": "knowledge_agent" not in state.parse_errors,
        "search_tool_used": any(c["tool"] == "search_kb" for c in state.tool_calls),
        "retrieval_hit": None if no_match else bool(retrieved & set(exp.get("sources", []))),
        "source_shown_in_log": None if no_match else bool(shown & set(exp.get("sources", []))),
        "coverage_decision": kb_says == (not no_match),
        "context_brief": bool(kb.get("context_brief")),
        "facts_verbatim": (all(_fact_in_passages(f, passages_text) for f in facts) if facts else no_match),
        "learned_reuse": bool(merged["resolved_ticket_matches"]) if exp.get("learned_reuse") else None,
        "scoped_to_product_line": _scoped(state),
    }


def _scoped(state):
    """Every retrieved passage belongs to the ticket's product line(s) or the general policies."""
    lines = [l for l in state.triage.get("product_lines", []) if l and l != "general"]
    if not lines:
        return None
    allowed = set(lines) | {"general"}
    passages = state.kb_merged["policy_passages"] + state.kb_merged["resolved_ticket_matches"]
    return all(p.get("scope", "general") in allowed for p in passages)


def check_resolution(state, exp, threshold: int) -> Dict[str, bool | None]:
    r = state.resolution
    reply = r.get("response", "")
    merged = state.kb_merged
    allowed = " ".join([p["text"] for p in merged["policy_passages"] + merged["resolved_ticket_matches"]]
                       + [state.ticket.complaint, state.ticket_id, state.ticket.order_id, state.ticket.customer_contact])
    allowed_nums = set(_numbers(allowed))
    invented = [n for n in _numbers(reply) if n not in allowed_nums]
    lower = reply.lower()
    no_match = exp.get("no_match", False)
    groups = exp.get("must_include", [])
    conf = r.get("confidence", -1)
    return {
        "json_valid": "resolution_agent" not in state.parse_errors,
        "numbers_grounded": not invented,
        "required_facts": all(any(p.lower() in lower for p in g) for g in groups) if groups else None,
        "honest_when_unknown": any(h in lower for h in HONEST) if no_match else None,
        "mentions_ticket": state.ticket_id in reply,
        "polite": any(w in lower for w in ["sorry", "thank", "hi ", "hello", "dear", "வணக்கம்", "நன்றி"]),
        "concise": len(reply.split()) <= 220,
        "citations_valid": (not state.grounding.get("invalid_citations")
                            and (bool(state.grounding.get("valid_citations")) if exp["status"] == "Resolved" else True))
        if state.grounding else None,
        "reply_grounded": bool(state.grounding.get("grounded")) if state.grounding else None,
        "confidence_calibrated": (conf < threshold) if no_match else (
            conf >= threshold if exp["status"] == "Resolved" else 0 <= conf <= 100),
        "_invented_numbers": invented,
    }


def check_log(state, exp, ticket_log) -> Dict[str, bool | None]:
    row = ticket_log.get(state.ticket_id) or {}
    status = row.get("status", "")
    return {
        "row_written": bool(row),
        "required_columns": bool(row) and not ticket_log.missing_required(row),
        "status_correct": status == exp["status"],
        "reason_when_escalated": bool(row.get("escalation_reason")) if status == "Escalated" else None,
        "logged_by_log_agent": state.logged_by == "log_agent",
        "kb_feed_correct": (row.get("fed_to_kb") == "yes") if status == "Resolved" else (row.get("fed_to_kb") == "no"),
        "agent_agrees_with_rules": bool(state.escalation.get("agreement")),
        "routed_to_correct_desk": (state.routing.get("name") == exp["team"]) if "team" in exp else None,
        "product_line_logged": bool(row.get("product_line")) and bool(row.get("assigned_team")),
    }


def check_orchestration(state, max_messages: int) -> Dict[str, bool | None]:
    return {
        "order_manager_query_kb_resolution_log": state.speaker_order == AGENT_ORDER,
        "one_division_group_ran": len({division_of(c["agent"]) for c in state.tool_calls
                                        if role_of(c["agent"]) in GROUP_ROLES}) == 1,
        "terminated_on_LOGGED": "LOGGED" in state.stop_reason,
        "within_message_cap": 0 < state.message_count <= max_messages + 4,
        "no_pipeline_error": not state.stop_reason.startswith("error"),
    }


# -------------------------------------------------------- structural checks
def structure_checks(settings) -> Dict[str, bool]:
    from agents.context import AgentContext
    from engine.offline_client import OfflineChatCompletionClient
    from engine.state import TicketState
    from engine.team import build_team
    from rag.knowledge import KnowledgeStore
    from ticketing.ticket_log import TicketLog

    ctx = AgentContext(settings=settings, state=TicketState("TKT-0000", TicketInput("x")),
                       store=KnowledgeStore(settings), ticket_log=TicketLog(settings.log_file, settings.data_dir))
    team, members = build_team(OfflineChatCompletionClient(), ctx)
    expected_tools = {MANAGER: {"identify_product_line"},
                      "query_agent": {"analyze_sentiment", "search_ticket_history"},
                      "knowledge_agent": {"search_kb"}, "resolution_agent": set(), "log_agent": {"log_ticket"}}
    divisions = [d for d in members if d != MANAGER]
    groups = team._participants[1:]
    out = {
        "support_manager + one agent group per division": len(divisions) == len(ctx.store.catalog.ids())
        and type(team).__name__ == "SelectorGroupChat" and len(groups) == len(divisions),
        "every group = 4 AutoGen AssistantAgents (query, knowledge, resolution, log)": all(
            len(members[d]) == 4 and all(type(a).__name__ == "AssistantAgent" for a in members[d]) for d in divisions)
        and all(type(g).__name__ in {"RoundRobinGroupChat", "SelectorGroupChat"} for g in groups),
    }
    sample = [members[MANAGER][0]] + members["general"]  # one representative of each role
    for a in sample:
        role = role_of(a.name)
        sys_msg = a._system_messages[0].content if a._system_messages else ""
        tools = {t.name for wb in a._workbench for t in getattr(wb, "_tools", [])}
        out[f"{role}: description + focused system message"] = bool(a.description) and len(sys_msg) > 400 and \
            "single responsibility" in sys_msg.lower()
        out[f"{role}: tools {sorted(expected_tools[role]) or 'none'}"] = tools == expected_tools[role]
    out["every group agent knows its division"] = all(
        ctx.store.catalog.name(d) in a._system_messages[0].content for d in divisions for a in members[d])
    def leaves(cond):
        subs = getattr(cond, "_conditions", None)
        return [x for c in subs for x in leaves(c)] if subs else [cond]

    kinds = {type(c).__name__: c for c in leaves(team._termination_condition)}
    logged = kinds.get("TextMentionTermination")
    out["desk stops on LOGGED, with source-match + message-cap safety nets"] = bool(logged) \
        and getattr(logged, "_termination_text", "") == "LOGGED" \
        and {"MaxMessageTermination", "SourceMatchTermination"} <= set(kinds)
    out["Support Manager guardrail can stop the desk (ROUTE: none)"] = "FunctionalTermination" in kinds
    return out


def channel_checks(settings) -> Dict[str, bool]:
    from channels.base import Outbox
    from channels.dispatcher import Dispatcher
    from channels.email_channel import parse_email, render_email_html
    from channels.inbox import Inbox
    from channels.telegram_channel import parse_update
    from channels.whatsapp_channel import normalize_number, parse_twilio_form
    from channels.base import InboundMessage

    wa = parse_twilio_form({"From": "whatsapp:+919800000009", "Body": "Where is my order?", "MessageSid": "SM1",
                            "ProfileName": "Asha"})
    tg = parse_update({"update_id": 11, "message": {"message_id": 5, "chat": {"id": 42}, "from": {"first_name": "Ravi"},
                                                    "text": "I forgot my password"}})
    raw = (b"From: Anita <anita@example.com>\r\nSubject: Broken blender\r\nMessage-ID: <x1@ex>\r\n"
           b"Content-Type: text/plain; charset=utf-8\r\n\r\nThe blender is broken.\r\n\r\nOn Mon, Bob wrote:\r\n> old\r\n")
    em = parse_email(raw, "1")
    html = render_email_html("Remya Industries Ltd", "TKT-0042", "Escalated", "Hi\n\n- step one\n- step two", "complaint")
    d = Dispatcher(settings, dry_run=True)
    results = [d.send_to_customer(ch, to, "TKT-0042", "Resolved", "Hello", "c") for ch, to in
               [("email", "a@example.com"), ("whatsapp", "+919800000009"), ("telegram", "42")]]
    inbox = Inbox(settings)
    first = inbox.add([InboundMessage(channel="whatsapp", sender_id="x", text="t", external_id="DUP-1")])
    second = inbox.add([InboundMessage(channel="whatsapp", sender_id="x", text="t", external_id="DUP-1")])
    return {
        "whatsapp_webhook_parsed": wa is not None and wa.sender_id == "whatsapp:+919800000009" and wa.sender_name == "Asha",
        "whatsapp_number_normalised": normalize_number("+91 98000 00009") == "whatsapp:+919800000009",
        "telegram_update_parsed": tg is not None and tg.sender_id == "42" and tg.text == "I forgot my password",
        "email_parsed_and_quote_stripped": em.sender_id == "anita@example.com" and em.text == "The blender is broken.",
        "email_html_template": "TKT-0042" in html and "ESCALATED" in html and "<li>step one</li>" in html,
        "reply_dispatched_to_3_channels": [r["status"] for r in results] == ["simulated"] * 3,
        "outbox_records_messages": len(Outbox(settings).read()) >= 3,
        "inbox_deduplicates": first == 1 and second == 0,
    }


def ui_checks() -> Dict[str, bool]:
    src = (BASE_DIR / "app.py").read_text(encoding="utf-8") + "".join(
        p.read_text(encoding="utf-8") for p in sorted((BASE_DIR / "ui").rglob("*.py")))
    return {
        "complaint text area + name/order id + submit": all(k in src for k in
                                                             ["st.text_area(\"Customer complaint\"", "Order ID", "form_submit_button"]),
        "each agent step shown live (st.status timeline)": "class LiveTimeline" in src and ".status(" in src,
        "final reply in highlighted box": "reply-box" in src,
        "green Resolved / red Escalated badge": "badge ok" in src and "badge bad" in src,
        "ticket log table + download (CSV & Excel)": "Download CSV" in src and "Download Excel" in src,
        "dashboard (category, status, urgency, sentiment, channel)": "by_category_status" in src and "over_time" in src,
        "channel inbox + outbox (email/telegram/whatsapp)": "Omni-channel inbox" in src and "Outbox" in src,
        "human-in-the-loop escalation queue": "Approve & send" in src,
        "product line / service selector + catalog tab": "Product line / service" in src and "Product lines & services" in src,
        "dashboard by product line and specialist desk": "by_product_line_status" in src and "desk_workload" in src,
        "citations + grounding shown with every reply": "Citations (" in src and "Grounded" in src,
        "retrieval trace (found by / ranks / rerank) and per-ticket trace viewer": "found by" in src and "read_trace" in src,
        "reply streamed live while it is written": "reply_chunk" in src,
        "follow-up linking field (conversation memory)": "Follow-up to ticket" in src,
        "ground-truth retrieval metrics shown": "retrieval_metrics.csv" in src,
        "app split into modules (app.py + ui/tabs)": len(list((BASE_DIR / "ui" / "tabs").glob("*.py"))) >= 8,
    }


def repo_checks() -> Dict[str, bool]:
    readme = (BASE_DIR / "README.md").read_text(encoding="utf-8") if (BASE_DIR / "README.md").exists() else ""
    gitignore = (BASE_DIR / ".gitignore").read_text(encoding="utf-8") if (BASE_DIR / ".gitignore").exists() else ""
    secret = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|gsk_[A-Za-z0-9]{20,}|\bAC[a-f0-9]{32}\b|"
                        r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b)")
    leaks = []
    for p in BASE_DIR.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".example", ".csv"} \
                and ".git" not in p.parts and p.name != ".env":
            if secret.search(p.read_text(encoding="utf-8", errors="ignore")):
                leaks.append(str(p.relative_to(BASE_DIR)))
    shots = [p for p in (BASE_DIR / "screenshots").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}]
    return {
        "README: setup, run, architecture, screenshots": all(
            h.lower() in readme.lower() for h in ["## Setup", "## Run", "## Architecture", "## Screenshots"]),
        "requirements.txt with pinned versions": (BASE_DIR / "requirements.txt").exists() and all(
            "==" in l for l in (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#") and not l.startswith("-")),
        "ground-truth retrieval set (>= 20 questions)": (BASE_DIR / "evaluation" / "retrieval_set.json").exists() and len(
            json.loads((BASE_DIR / "evaluation" / "retrieval_set.json").read_text(encoding="utf-8"))["questions"]) >= 20,
        ".env.example present": (BASE_DIR / ".env.example").exists(),
        ".env ignored by git": bool(re.search(r"^\.env$", gitignore, flags=re.M)),
        "no API keys in code": not leaks,
        "one-command start (streamlit run app.py)": (BASE_DIR / "app.py").exists() and "streamlit run app.py" in readme,
        "screenshots folder (>= 3 images)": len(shots) >= 3,
        "tests folder": any((BASE_DIR / "tests").glob("test_*.py")),
        **_split_checks(),
    }


def _split_checks() -> Dict[str, bool]:
    from evaluation import retrieval_eval

    try:
        questions = retrieval_eval.load_set()
    except (OSError, ValueError, KeyError):
        return {"held-out retrieval split unchanged since it was frozen": False}
    frozen = retrieval_eval.frozen_fingerprint()
    return {"held-out retrieval split unchanged since it was frozen (SHA-256)":
                bool(frozen) and frozen == retrieval_eval.split_fingerprint(questions),
            "no held-out question copied into the code or the knowledge base":
                not retrieval_eval.leaked_questions(questions)}


# ------------------------------------------------------------------- runner
async def run(provider: str | None, model: str | None, orchestration: str | None, only: List[str],
              judge: bool = True, embedding: str | None = None, embedding_model: str | None = None) -> Dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = RUNS_DIR / run_id
    work.mkdir(parents=True, exist_ok=True)
    overrides = {"data_dir": work / "data", "log_file": work / "tickets.csv",
                 "learned_file": work / "learned_solutions.jsonl"}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        overrides["llm_model"] = model
    if orchestration:
        overrides["orchestration_mode"] = orchestration
    retrieval_overrides = {k: v for k, v in overrides.items() if k in {"llm_provider", "llm_model"}}
    if embedding:
        overrides["embedding_backend"] = retrieval_overrides["embedding_backend"] = embedding
    if embedding_model:
        overrides["embedding_model"] = retrieval_overrides["embedding_model"] = embedding_model
    settings = get_settings(**overrides)

    from channels.dispatcher import Dispatcher

    engine = SupportEngine(settings, dispatcher=Dispatcher(settings, dry_run=True))
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]

    case_rows, node_rows = [], []
    faith_rows: List[Dict] = []
    judge_inputs: List[Dict] = []
    node_checks: Dict[str, List[Dict]] = {"Guardrails": [], "Support Manager": [], "Customer Query Agent": [],
                                          "Knowledge Base Agent": [], "Resolution Agent": [], "Log Agent": [],
                                          "Orchestration": []}
    skipped = []
    for case in cases:
        if case.get("llm_only") and settings.is_offline:
            skipped.append(case["id"])
            continue
        exp = case["expect"]
        t0 = time.time()
        result = await engine.process(TicketInput(complaint=case["complaint"], customer_name=case.get("customer_name", ""),
                                                  order_id=case.get("order_id", ""), channel=case.get("channel", "web"),
                                                  customer_contact=case.get("customer_contact", "")))
        st = result.state
        if exp["status"] == "Filtered" or result.status == "Filtered":
            row = engine.ticket_log.get(st.ticket_id) or {}
            checks = {"Guardrails": check_guardrail(st, exp), "Support Manager": {}, "Customer Query Agent": {},
                      "Knowledge Base Agent": {}, "Resolution Agent": {"_invented_numbers": []},
                      "Log Agent": {"row_written": bool(row),
                                    "required_columns": bool(row) and not engine.ticket_log.missing_required(row),
                                    "status_correct": row.get("status") == exp["status"]},
                      "Orchestration": {}}
        else:
            checks = {
                "Guardrails": check_guardrail(st, exp),
                "Support Manager": check_manager(st, exp),
                "Customer Query Agent": check_query(st, exp),
                "Knowledge Base Agent": check_knowledge(st, exp),
                "Resolution Agent": check_resolution(st, exp, settings.confidence_threshold),
                "Log Agent": check_log(st, exp, engine.ticket_log),
                "Orchestration": check_orchestration(st, settings.max_messages),
            }
        invented = checks["Resolution Agent"].pop("_invented_numbers")
        for node, c in checks.items():
            node_checks[node].append(c)
            for name, val in c.items():
                node_rows.append({"case": case["id"], "node": node, "check": name,
                                  "result": "n/a" if val is None else ("pass" if val else "FAIL")})
        case_rows.append({
            "case": case["id"], "title": case["title"], "channel": case.get("channel", "web"),
            "product_line": st.routing.get("product_line", ""), "desk": st.routing.get("name", ""),
            "category": st.triage.get("category"), "urgency": st.triage.get("urgency"),
            "sentiment": st.triage.get("sentiment"), "kb_sources": st.log_row.get("kb_sources"),
            "confidence": st.resolution.get("confidence"), "status": result.status, "expected_status": exp["status"],
            "status_ok": result.status == exp["status"],
            "reasons": st.log_row.get("escalation_reason", ""),
            "group": st.manager_route.get("group", "") or ("(stopped)" if result.status == "Filtered" else ""),
            "guard": _ratio(checks["Guardrails"]),
            "manager": _ratio(checks["Support Manager"]),
            "query": _ratio(checks["Customer Query Agent"]), "knowledge": _ratio(checks["Knowledge Base Agent"]),
            "resolution": _ratio(checks["Resolution Agent"]), "log": _ratio(checks["Log Agent"]),
            "orchestration": _ratio(checks["Orchestration"]),
            "seconds": round(time.time() - t0, 2), "invented_numbers": ", ".join(invented),
            "reply": result.reply,
        })
        if result.status != "Filtered" and st.resolution.get("response"):
            g = st.grounding or {}
            merged = st.kb_merged
            faith_rows.append({"case": case["id"], "status": result.status,
                               "grounded": bool(g.get("grounded")), "supported_ratio": g.get("supported_ratio"),
                               "unsupported_figures": ", ".join(g.get("unsupported_figures", [])),
                               "citations": ", ".join(g.get("valid_citations", [])),
                               "invalid_citations": ", ".join(g.get("invalid_citations", [])),
                               "unsupported_sentences": " | ".join(g.get("unsupported_sentences", []))})
            judge_inputs.append({"case": case["id"], "complaint": case["complaint"], "reply": st.resolution["response"],
                                 "passages": merged["policy_passages"] + merged["resolved_ticket_matches"]})
        print(f"{case['id']:6} {result.status:9} (expected {exp['status']:9}) -> {st.manager_route.get('group', ''):26} "
              f"G={case_rows[-1]['guard']:.2f} M={case_rows[-1]['manager']:.2f} Q={case_rows[-1]['query']:.2f} K={case_rows[-1]['knowledge']:.2f} "
              f"R={case_rows[-1]['resolution']:.2f} L={case_rows[-1]['log']:.2f} O={case_rows[-1]['orchestration']:.2f}")

    # ---- answer faithfulness: LLM judge (real LLM only) on top of the deterministic grounding check
    judge_used = bool(judge and not settings.is_offline and judge_inputs)
    if judge_used:
        from engine.llm import build_model_client
        from evaluation.judge import judge_reply

        print(f"\nLLM-as-judge: scoring {len(judge_inputs)} replies for faithfulness and relevance...")
        client = build_model_client(settings)
        try:
            for item, row in zip(judge_inputs, faith_rows):
                verdict = await judge_reply(client, item["complaint"], item["passages"], item["reply"])
                row.update({"judge_faithfulness": verdict["faithfulness"], "judge_relevance": verdict["relevance"],
                            "judge_unsupported": " | ".join(map(str, verdict["unsupported_claims"]))[:300],
                            "judge_comment": verdict["comment"]})
        finally:
            await client.close()
    scored = [r for r in faith_rows if r.get("judge_faithfulness")]
    faithfulness = {
        "replies": len(faith_rows),
        "grounded_share": round(statistics.mean(r["grounded"] for r in faith_rows), 3) if faith_rows else None,
        "lexical_support": round(statistics.mean(r["supported_ratio"] for r in faith_rows
                                                 if r["supported_ratio"] is not None), 3) if faith_rows else None,
        "replies_with_unsupported_figures": sum(1 for r in faith_rows if r["unsupported_figures"]),
        "judge": "LLM-as-judge (" + describe(settings) + ")" if judge_used else "not run (offline model)",
        "judge_faithfulness_avg": round(statistics.mean(r["judge_faithfulness"] for r in scored), 2) if scored else None,
        "judge_relevance_avg": round(statistics.mean(r["judge_relevance"] for r in scored if r.get("judge_relevance")),
                                     2) if scored else None,
    }

    # ---- retrieval ground truth (hit-rate@k, MRR, abstention, ablation) - same knowledge base, no LLM needed
    from evaluation import retrieval_eval

    retrieval = await retrieval_eval.evaluate_async(get_settings(**retrieval_overrides), write=True)
    held_out = retrieval["production"]["held-out test"]

    # ---- aggregate node scores
    def agg(node: str) -> Dict:
        per_check: Dict[str, List[bool]] = {}
        for c in node_checks[node]:
            for k, v in c.items():
                if v is not None:
                    per_check.setdefault(k, []).append(bool(v))
        rates = {k: sum(v) / len(v) for k, v in per_check.items()}
        return {"node": node, "score": round(statistics.mean(rates.values()), 3) if rates else 0.0,
                "checks": {k: f"{sum(per_check[k])}/{len(per_check[k])}" for k in per_check}}

    nodes = [agg(n) for n in node_checks]
    structure = structure_checks(settings)
    channels = channel_checks(settings)
    ui = ui_checks()
    repo = repo_checks()
    nodes.append({"node": "Channels (Email / Telegram / WhatsApp)", "score": _ratio(channels),
                  "checks": {k: "pass" if v else "FAIL" for k, v in channels.items()}})

    # escalation confusion matrix
    case_rows_all = case_rows
    case_rows = [r for r in case_rows_all if r["expected_status"] != "Filtered"]  # escalation matrix: real tickets
    tp = sum(1 for r in case_rows if r["expected_status"] == "Escalated" and r["status"] == "Escalated")
    fn = sum(1 for r in case_rows if r["expected_status"] == "Escalated" and r["status"] != "Escalated")
    fp = sum(1 for r in case_rows if r["expected_status"] == "Resolved" and r["status"] == "Escalated")
    tn = sum(1 for r in case_rows if r["expected_status"] == "Resolved" and r["status"] == "Resolved")
    esc_acc = (tp + tn) / max(len(case_rows), 1)
    case_rows = case_rows_all

    by = {n["node"]: n["score"] for n in nodes}
    res_checks = [c for c in node_checks["Resolution Agent"] if "numbers_grounded" in c]  # real tickets only
    grounded = statistics.mean(c["numbers_grounded"] for c in res_checks) if res_checks else 0
    honest = [c["honest_when_unknown"] for c in res_checks if c["honest_when_unknown"] is not None]
    json_ok = statistics.mean(
        statistics.mean([c["json_valid"] for c in node_checks[n] if "json_valid" in c] or [0]) for n in
        ["Customer Query Agent", "Knowledge Base Agent", "Resolution Agent"]) if case_rows else 0
    areas = [
        ("Agent design", 20, statistics.mean([_ratio(structure), json_ok, by["Support Manager"],
                                              by["Customer Query Agent"]]),
         "Support Manager + one 4-agent group per division, single-responsibility prompts and tools; JSON contract; routing and triage accuracy"),
        ("Knowledge grounding", 20, statistics.mean([by["Knowledge Base Agent"], grounded,
                                                     statistics.mean(honest) if honest else 1.0,
                                                     faithfulness["grounded_share"] or 0.0,
                                                     held_out["hit@3"], held_out["abstention accuracy"]]),
         "agent checks, no invented numbers, honest 'not covered' answers, grounded replies, ground-truth "
         "hit@3 and abstention on the HELD-OUT questions"),
        ("Workflow & escalation", 20, statistics.mean([by["Orchestration"], esc_acc, by["Guardrails"]]),
         "Manager → division group (Query → Knowledge → Resolution → Log), LOGGED termination + safety nets, escalation accuracy, guardrails"),
        ("Logging", 15, by["Log Agent"], "row per ticket, required columns, status/reason, KB feed rule"),
        ("Streamlit UI", 15, _ratio(ui), "feature checklist (see screenshots for the visual evidence)"),
        ("Code & README", 10, _ratio(repo), "structure, no secrets, setup steps, screenshots, tests"),
    ]
    rubric = {"areas": [{"area": a, "max": m, "score": round(m * s, 1), "pct": f"{s:.0%}", "evidence": e}
                        for a, m, s, e in areas]}
    rubric["total"] = round(sum(x["score"] for x in rubric["areas"]), 1)

    report = {
        "meta": {"run_id": run_id, "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": describe(settings),
                 "orchestration": settings.orchestration_mode, "cases_run": len(case_rows), "skipped": skipped,
                 "confidence_threshold": settings.confidence_threshold, "workspace": str(work.relative_to(BASE_DIR))},
        "rubric": rubric,
        "nodes": [{"node": n["node"], "score": f"{n['score']:.0%}",
                   "checks": "; ".join(f"{k} {v}" for k, v in n["checks"].items())} for n in nodes],
        "escalation": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "accuracy": round(esc_acc, 3)},
        "faithfulness": faithfulness,
        "retrieval": {"questions": retrieval["questions"], "languages": retrieval["languages"],
                      "embedding": retrieval["embedding"], "reranker": retrieval["reranker"],
                      "llm": retrieval["llm"], "production_variant": held_out["variant"],
                      "integrity": retrieval["integrity"], "summary": retrieval["summary"]},
        "structure": structure, "channels": channels, "ui": ui, "repo": repo,
        "cases": [{k: v for k, v in r.items() if k != "reply"} for r in case_rows],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(case_rows).to_csv(REPORT_DIR / "case_results.csv", index=False)
    pd.DataFrame(node_rows).to_csv(REPORT_DIR / "node_checks.csv", index=False)
    pd.DataFrame(faith_rows).to_csv(REPORT_DIR / "faithfulness.csv", index=False)
    if Path(settings.log_file).exists():
        shutil.copy(settings.log_file, REPORT_DIR / "eval_tickets.csv")
    (REPORT_DIR / "EVALUATION.md").write_text(render_markdown(report, case_rows, node_rows), encoding="utf-8")
    print(f"\nRubric estimate: {rubric['total']}/100  ->  {REPORT_DIR / 'EVALUATION.md'}")
    return report


# ------------------------------------------------------------------ report
def retrieval_markdown(rep: Dict) -> List[str]:
    """Ground-truth retrieval and answer-faithfulness sections of EVALUATION.md."""
    out: List[str] = []
    rv = rep.get("retrieval")
    if rv:
        integ = rv.get("integrity") or {}
        th = integ.get("thresholds") or {}
        out += ["### Retrieval against hand-labelled ground truth", "",
                f"{rv['questions']} questions ({', '.join(rv['languages'])}) in `evaluation/retrieval_set.json`, each "
                "labelled with the passage(s) that answer it, plus questions the documents do not cover. The *dev* "
                "split was used to tune query expansion, the reranker and the thresholds; the *held-out test* split "
                f"is never tuned on and is the honest number. Embeddings: `{rv['embedding']}`, reranker: "
                f"`{rv['reranker']}`, LLM: `{rv.get('llm', 'none')}`; production path: **{rv.get('production_variant', '')}**.",
                ""]
        if integ:
            out += [f"- Held-out split SHA-256 `{integ.get('test_split_sha256', '')[:16]}…` — "
                    + ("**unchanged** since it was frozen on " + integ.get("frozen_at", "")
                       if integ.get("unchanged_since_freeze") else "**changed since it was frozen**"),
                    "- Held-out questions found in the code or knowledge base: "
                    + (", ".join(integ.get("leaked_into_code") or []) or "none"),
                    f"- 'Covered' thresholds: min relevance {th.get('min_relevance')}, min coverage "
                    f"{th.get('min_coverage')}, strong relevance {th.get('strong_relevance')} — {th.get('source', '')}", ""]
        cols = ["variant", "hit@1", "hit@3", "hit@3 95% CI", "hit@5", "MRR@10", "hit@3 Tamil/Hindi",
                "abstention accuracy", "abstention 95% CI", "false 'covered' on unanswerable", "missed answerable"]
        for split in ("held-out test", "dev (tuning)"):
            rows = [r for r in rv["summary"] if r["split"] == split]
            out += [f"**{split}**", "", "| " + " | ".join(cols) + " |", "|---" + "|---:" * (len(cols) - 1) + "|"]
            out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
            out.append("")
        out += ["hit@k = share of answerable questions whose labelled passage is in the top k; MRR@10 = mean of "
                "1/rank of the first labelled passage; abstention accuracy = covered/not-covered decision correct "
                "(answerable must be covered, unanswerable must not); 95% CI = bootstrap interval (2,000 resamples) - "
                "with ~30 held-out questions one question moves hit@3 by about 3.5 points, so read the interval, not "
                "the third decimal. Per-question results: `evaluation/reports/retrieval_cases.csv`.", ""]
    f = rep.get("faithfulness")
    if f:
        out += ["### Answer faithfulness", "",
                "| Measure | Value |", "|---|---:|",
                f"| Replies checked | {f['replies']} |",
                f"| Grounded (every figure found in the retrieved passages, valid citations) | {f['grounded_share']:.0%} |"
                if f["grounded_share"] is not None else "| Grounded | - |",
                f"| Lexical claim support (factual sentences whose key words are in the passages) | "
                f"{f['lexical_support']:.0%} |" if f["lexical_support"] is not None else "| Lexical support | - |",
                f"| Replies with unsupported figures | {f['replies_with_unsupported_figures']} |",
                f"| LLM-as-judge faithfulness (1-5) | {f['judge_faithfulness_avg'] or '-'} |",
                f"| LLM-as-judge relevance (1-5) | {f['judge_relevance_avg'] or '-'} |", "",
                f"Judge: {f['judge']}. Per-reply results: `evaluation/reports/faithfulness.csv`.", ""]
    return out


def render_markdown(rep: Dict, case_rows: List[Dict], node_rows: List[Dict]) -> str:
    m, r = rep["meta"], rep["rubric"]
    lines = [
        "# Evaluation Report — AI Customer Support Automation",
        "",
        f"*Model:* **{m['model']}** · *Orchestration:* **{m['orchestration']}** · *Run:* {m['run_at']} · "
        f"*Cases:* {m['cases_run']}" + (f" (skipped for this model: {', '.join(m['skipped'])})" if m["skipped"] else "")
        + f" · *Confidence threshold:* {m['confidence_threshold']}%",
        "",
        "> Automated self-assessment produced by `python -m evaluation.evaluate`. Each test case runs through the real",
        "> AutoGen engine in an isolated workspace; channel delivery is dry-run. The official grade is the evaluator's.",
        "",
    ]
    if m["model"].startswith("Offline"):
        lines += [
            "> **Offline demo model.** This run used the rule-based stand-in, whose rules were written alongside these",
            "> test cases. A perfect score here proves the plumbing (tools, team, termination, logging, escalation,",
            "> channels) works end to end; it does not measure LLM quality. Re-run with your LLM provider before submitting.",
            "",
        ]
    lines += [
        f"## 1. Rubric summary — {r['total']:.0f} / 100",
        "",
        "| Area | Marks | Estimate | How it is measured |",
        "|---|---:|---:|---|",
    ]
    for a in r["areas"]:
        lines.append(f"| {a['area']} | {a['max']} | **{a['score']}** ({a['pct']}) | {a['evidence']} |")
    lines += ["", "## 2. Node-by-node scores", "", "| Node | Score | Checks (passed / applicable) |", "|---|---:|---|"]
    for n in rep["nodes"]:
        lines.append(f"| {n['node']} | **{n['score']}** | {n['checks']} |")
    e = rep["escalation"]
    lines += ["", "### Escalation decision (confusion matrix)", "",
              "| | Predicted Escalated | Predicted Resolved |", "|---|---:|---:|",
              f"| **Expected Escalated** | {e['tp']} | {e['fn']} |",
              f"| **Expected Resolved** | {e['fp']} | {e['tn']} |", "",
              f"Escalation accuracy: **{e['accuracy']:.0%}**", "",
              ""]
    lines += retrieval_markdown(rep)
    lines += ["## 3. Test cases", "",
              "| Case | Scenario | Channel | Manager → group | Product line → desk | Category · Urgency · Sentiment | KB source | Conf. | Status (expected) | G | M | Q | K | R | L | O |",
              "|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in case_rows:
        mark = "✅" if c["status_ok"] else "❌"
        lines.append(f"| {c['case']} | {c['title']} | {c['channel']} | {c['group']} | {c['product_line']} → {c['desk']} | "
                     f"{c['category']} · {c['urgency']} · {c['sentiment']} | "
                     f"{c['kb_sources']} | {c['confidence']} | {mark} {c['status']} ({c['expected_status']}) | "
                     f"{c['guard']:.0%} | {c['manager']:.0%} | {c['query']:.0%} | {c['knowledge']:.0%} | {c['resolution']:.0%} | {c['log']:.0%} | {c['orchestration']:.0%} |")
    fails = [r for r in node_rows if r["result"] == "FAIL"]
    lines += ["", "G = Guardrails, M = Support Manager, Q = Customer Query Agent, K = Knowledge Base Agent, R = Resolution Agent, L = Log Agent, O = Orchestration.",
              "", "### Failed checks", ""]
    lines += [f"- **{f['case']}** · {f['node']} · `{f['check']}`" for f in fails] or ["- none"]
    lines += ["", "## 4. What each node is checked for", "",
              "| Node | Checks |", "|---|---|",
              "| Guardrails | messages that are not tickets (greeting, spam, injection, off-topic, gibberish, duplicate) are stopped by the expected stage with the right reason and no agent group runs; real complaints are never stopped; injection attempts flagged; card numbers / OTPs never reach the log; no AI chatter in replies |",
              "| Support Manager | valid JSON; `identify_product_line` called; `ROUTE:` line given; routed to the correct division's agent group; *needs human* raised when no division or document fits; secondary divisions of a multi-division ticket listed; reason + brief for the group; the selector handed the ticket to exactly that group |",
              "| Customer Query Agent | valid JSON; category / urgency / sentiment match the expected labels; `analyze_sentiment` and `search_ticket_history` (history RAG) called; correct product line / service; every product line of a multi-line ticket found; one-line summary; repeat complaint detected; language detected |",
              "| Knowledge Base Agent | valid JSON; `search_kb` called; expected policy file retrieved (hit@k) and shown in the log; correct covered / not-covered decision; context brief for the Resolution Agent; every key fact is verbatim from a retrieved passage; learned solution re-used; retrieval stayed inside the ticket's product line(s) + general policies |",
              "| Resolution Agent | valid JSON; every number in the reply exists in the retrieved passages or the ticket (no invented policy); required policy facts present; honest answer when the KB has nothing; ticket number; polite; ≤ 220 words; confidence calibrated against the threshold; citations point to retrieved passages; reply passes the runtime grounding check |",
              "| Log Agent | one CSV row per ticket; all required columns filled; status matches the expected result; escalation reason present; recorded through the `log_ticket` tool (not the engine fallback); only resolved, confident, policy-backed answers fed to the KB; agent's own decision agrees with the rule engine; routed to the correct specialist desk; product line + desk logged |",
              "| Orchestration | speaker order Manager → Query → Knowledge → Resolution → Log; exactly one division group ran; stopped by `TextMentionTermination('LOGGED')`; within the message cap; no pipeline error |",
              "| Channels | Twilio WhatsApp webhook parsing + number normalisation; Telegram update parsing; email parsing with quoted-reply stripping; HTML template mail; reply routed to email / WhatsApp / Telegram; outbox audit trail; inbox de-duplication |",
              "", "## 5. Structural, UI and repository checks", ""]
    for title, block in [("Agent structure", rep["structure"]), ("Channels", rep["channels"]),
                         ("Streamlit UI", rep["ui"]), ("Code & README", rep["repo"])]:
        lines.append(f"**{title}**")
        lines.append("")
        lines += [f"- {'✅' if v else '❌'} {k}" for k, v in block.items()]
        lines.append("")
    lines += ["## 6. Reproduce", "", "```bash", "python -m evaluation.evaluate                     # provider from .env",
              "python -m evaluation.evaluate --provider offline  # no API key",
              "python -m evaluation.evaluate --embedding openai --embedding-model text-embedding-3-small",
              "python -m evaluation.retrieval_eval               # retrieval only (ablation, CIs, integrity)",
              "python -m evaluation.retrieval_eval --calibrate   # re-fit the 'covered' thresholds on the dev split", "```", "",
              f"Raw results: `evaluation/reports/case_results.csv`, `node_checks.csv`, `eval_tickets.csv` "
              f"(workspace `{m['workspace']}`)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", help="openai | azure | gemini | groq | ollama | offline (default: .env)")
    ap.add_argument("--model")
    ap.add_argument("--orchestration", choices=["round_robin", "selector"])
    ap.add_argument("--cases", default="", help="comma-separated case ids, e.g. PDF-1,PDF-5")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge faithfulness scoring")
    ap.add_argument("--embedding", choices=["tfidf", "openai", "sentence-transformers"],
                    help="embedding backend for the run (default: .env EMBEDDING_BACKEND)")
    ap.add_argument("--embedding-model", help="embedding model, e.g. text-embedding-3-small")
    args = ap.parse_args()
    import logging

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args.provider, args.model, args.orchestration, [c for c in args.cases.split(",") if c],
                    judge=not args.no_judge, embedding=args.embedding, embedding_model=args.embedding_model))


if __name__ == "__main__":
    main()
