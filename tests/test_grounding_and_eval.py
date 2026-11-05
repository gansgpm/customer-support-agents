"""Grounding verifier, 'covered' threshold precedence and the integrity checks of the retrieval evaluation."""
import asyncio
import json

from engine import escalation, grounding
from engine.offline_client import OfflineChatCompletionClient
from evaluation import retrieval_eval
from rag import knowledge
from rag.knowledge import DEFAULT_THRESHOLDS, KnowledgeStore, is_covered

PASSAGES = [{"ref": "S1", "source": "refund_policy.md", "section": "Refund timelines",
             "text": "Refunds are credited to the original payment method within 5-7 business days after the "
                     "returned item passes the quality check."}]


class FakeClient:
    """Stands in for an LLM: returns a fixed answer and records the prompt."""

    def __init__(self, content):
        self.content, self.prompts = content, []

    async def create(self, messages, **_):
        self.prompts.append(messages)
        if isinstance(self.content, Exception):
            raise self.content
        return type("Result", (), {"content": self.content})()


# ------------------------------------------------------------------ LLM claim verifier
def test_llm_verifier_flags_claims_the_passages_do_not_support():
    client = FakeClient('```json\n{"unsupported_claims": ["You will also get a 10% voucher."]}\n```')
    verdict = asyncio.run(grounding.llm_verify(client, "Your refund arrives in 5-7 business days. You will also "
                                                       "get a 10% voucher.", PASSAGES))
    assert verdict["ran"] is True and verdict["unsupported_claims"] == ["You will also get a 10% voucher."]
    prompt = client.prompts[0][1].content
    assert "PASSAGES:" in prompt and "refund_policy.md" in prompt and "REPLY:" in prompt

    merged = grounding.merge_llm_verdict({"grounded": True, "note": "grounded"}, verdict)
    assert merged["grounded"] is False and merged["unsupported_claims"] == verdict["unsupported_claims"]
    assert merged["note"].startswith("LLM verifier: 1 unsupported claim")


def test_llm_verifier_clean_reply_stays_grounded_and_errors_never_block():
    ok = asyncio.run(grounding.llm_verify(FakeClient('{"unsupported_claims": []}'), "Refunds take 5-7 days.", PASSAGES))
    merged = grounding.merge_llm_verdict({"grounded": True, "note": "grounded"}, ok)
    assert merged["grounded"] is True and merged["note"] == "grounded"

    failed = asyncio.run(grounding.llm_verify(FakeClient(TimeoutError("slow")), "Refunds take 5-7 days.", PASSAGES))
    assert failed["ran"] is False and "TimeoutError" in failed["error"]
    assert grounding.merge_llm_verdict({"grounded": True}, failed)["grounded"] is True


def test_unsupported_claims_send_the_ticket_to_a_human(settings):
    triage = {"category": "Refund", "urgency": "Low", "sentiment": "Neutral", "risk_flags": []}
    resolution = {"confidence": 95, "response": "ok", "answer_found": True}
    clean = escalation.decide(triage, {"match_found": True}, resolution, settings, grounding={"grounded": True})
    flagged = escalation.decide(triage, {"match_found": True}, resolution, settings,
                                grounding={"grounded": False, "unsupported_claims": ["a 10% voucher"]})
    assert clean["status"] == "Resolved"
    assert flagged["status"] == "Escalated"
    assert any("claim" in r for r in flagged["reasons"])


def test_verifier_runs_only_with_a_real_llm_unless_forced(settings):
    assert not grounding.verifier_enabled(settings)  # offline + auto
    assert grounding.verifier_enabled(settings.with_overrides(grounding_verifier="llm"))
    assert grounding.verifier_enabled(settings.with_overrides(llm_provider="openai"))
    assert not grounding.verifier_enabled(settings.with_overrides(llm_provider="openai", grounding_verifier="off"))


def test_offline_client_answers_the_verifier_prompt():
    client = OfflineChatCompletionClient()
    verdict = asyncio.run(grounding.llm_verify(
        client, "Refunds are credited within 5-7 business days. We will also send a free gift hamper to your home.",
        PASSAGES))
    assert verdict["ran"] is True
    assert any("gift hamper" in c for c in verdict["unsupported_claims"])
    assert not any("credited" in c for c in verdict["unsupported_claims"])


# ------------------------------------------------------------------ 'covered' thresholds
def test_is_covered_needs_relevance_and_coverage_or_a_very_close_match():
    th = {"min_relevance": 0.2, "min_coverage": 0.4, "strong_relevance": 0.5}
    assert is_covered(0.3, 0.5, th)
    assert not is_covered(0.1, 1.0, th)            # not relevant enough, whatever the word overlap
    assert not is_covered(0.3, 0.3, th)            # relevant but most key words missing
    assert is_covered(0.6, 0.3, th)                # very close match, some key words
    assert not is_covered(0.6, 0.1, th)            # ... but not with almost none


def test_threshold_precedence_env_then_calibration_then_default(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "CALIBRATION_DIR", tmp_path / "calibration")
    store = KnowledgeStore(settings)
    th = store.thresholds()
    assert th["source"] == "default for tfidf"
    assert th["min_relevance"] == DEFAULT_THRESHOLDS["tfidf"]["min_relevance"]

    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "tfidf.json").write_text(json.dumps(
        {"min_relevance": 0.15, "min_coverage": 0.33, "strong_relevance": 0.3, "calibrated_at": "today"}))
    th = KnowledgeStore(settings).thresholds()
    assert (th["min_relevance"], th["min_coverage"], th["strong_relevance"]) == (0.15, 0.33, 0.3)
    assert th["source"].startswith("calibrated (tfidf.json")

    th = KnowledgeStore(settings.with_overrides(kb_min_relevance=0.5)).thresholds()
    assert th["min_relevance"] == 0.5 and th["min_coverage"] == 0.33 and th["source"] == "set in .env"


def test_auto_and_numbers_in_env(monkeypatch):
    from engine.config import _env_threshold

    for raw, expected in [("auto", None), ("", None), ("0.25", 0.25), ("nonsense", None)]:
        monkeypatch.setenv("KB_MIN_RELEVANCE", raw)
        assert _env_threshold("KB_MIN_RELEVANCE") == expected


# ------------------------------------------------------------------ evaluation integrity
def test_held_out_split_is_frozen_and_not_copied_into_the_code():
    questions = retrieval_eval.load_set()
    test = [q for q in questions if q.get("split") == "test"]
    dev = [q for q in questions if q.get("split") == "dev"]
    assert len(test) >= 25 and len(dev) >= 40
    assert not {q["id"] for q in test} & {q["id"] for q in dev}
    assert any(not q.get("answerable", True) for q in test), "the held-out split must test abstention too"
    assert retrieval_eval.split_fingerprint(questions) == retrieval_eval.frozen_fingerprint()
    assert retrieval_eval.leaked_questions(questions) == []


def test_fingerprint_changes_when_a_held_out_label_is_edited():
    questions = retrieval_eval.load_set()
    before = retrieval_eval.split_fingerprint(questions)
    edited = [dict(q) for q in questions]
    first_test = next(q for q in edited if q.get("split") == "test")
    first_test["expected"] = [["some_other_file.md", "Anything"]]
    assert retrieval_eval.split_fingerprint(edited) != before
    # editing a DEV question does not touch the held-out fingerprint
    dev_edit = [dict(q) for q in questions]
    next(q for q in dev_edit if q.get("split") == "dev")["question"] = "changed"
    assert retrieval_eval.split_fingerprint(dev_edit) == before


class FakeHelper:
    """LLMHelper stand-in: rewrites nothing, says 'covered' only when the top passage is the labelled one."""
    model, calls = "fake-llm", 0

    def __init__(self, labels):
        self.labels = labels

    async def rewrite(self, question):
        self.calls += 1
        return question

    async def verdict(self, question, passages):
        self.calls += 1
        return bool(self.labels.get(question))


def test_retrieval_eval_variants_and_metrics_with_a_fake_llm(settings):
    questions = retrieval_eval.load_set()[:12]
    store = KnowledgeStore(retrieval_eval.isolated(settings))
    names = [v["name"] for v in retrieval_eval.variants(settings, llm=True)]
    assert sum(1 for v in retrieval_eval.variants(settings, llm=True) if v.get("production")) == 1
    assert "LLM coverage verdict (production)" in names[-1]
    prod = retrieval_eval.variants(settings, llm=True)[-1]
    helper = FakeHelper({q["question"]: q.get("answerable", True) for q in questions})
    rows = asyncio.run(retrieval_eval.run_variant(store, questions, prod, helper))
    assert helper.calls >= len(questions)
    assert all(r["abstention_ok"] for r in rows)  # the (perfect) fake verdict decides 'covered'
    summary = retrieval_eval.summarise(rows)
    row = next(s for s in summary if s["split"] == "all")
    assert row["abstention accuracy"] == 1.0 and row["false 'covered' on unanswerable"] == 0
    lo, hi = map(float, row["hit@3 95% CI"].split("-"))
    assert 0.0 <= lo <= row["hit@3"] <= hi <= 1.0


# ------------------------------------------------------------------ fixes after the first real-LLM run
def test_numbered_list_is_not_a_figure():
    reply = ("A promo code can fail for these reasons:\n1. The code has expired.\n2) Your cart is below the minimum "
             "order value.\n3. Only one promo code per order.\nRefunds take 5-7 business days.")
    g = grounding.check(reply, ["S1"], {"S1": PASSAGES[0]}, PASSAGES)
    assert g["unsupported_figures"] == []
    assert grounding.numbers("Pay Rs. 499 within 2 days") == ["499", "2"]


def test_verifier_sees_the_ticket_and_overrules_copied_claims():
    passages = [{"ref": "S1", "source": "electronics.md", "section": "Warranty",
                 "text": "Mobile phones, laptops, tablets and other electronics come with a 1-year manufacturer "
                         "warranty. After the 10-day return window, warranty repairs are handled by the brand's "
                         "authorised service centre."}]
    answer = json.dumps({"claims": [
        {"claim": "Your laptop comes with a 1-year manufacturer warranty", "verdict": "unsupported", "evidence": ""},
        {"claim": "Your order ORD-4512 is 10 days late", "verdict": "unsupported", "evidence": ""},
        {"claim": "We will also give you a free replacement laptop today", "verdict": "unsupported", "evidence": ""},
        {"claim": "Repairs are done by the authorised service centre", "verdict": "supported", "evidence": "..."}]})
    client = FakeClient(answer)
    verdict = asyncio.run(grounding.llm_verify(client, "reply text", passages,
                                               customer_text="My order is 10 days late and my laptop is dead.",
                                               ticket_details="ticket TKT-0007; order ORD-4512", cited=["S1"]))
    prompt = client.prompts[0][1].content
    assert "TICKET:" in prompt and "My order is 10 days late" in prompt and "ORD-4512" in prompt
    assert verdict["unsupported_claims"] == ["We will also give you a free replacement laptop today"]
    assert len(verdict["overruled"]) == 2 and verdict["checked_claims"] == 4


def test_near_verbatim_keeps_negations_and_figures_apart():
    source = ["Delivery is free for orders of Rs. 499 or more."]
    assert grounding.near_verbatim("Delivery is free for orders of Rs. 499 or more", source)
    assert not grounding.near_verbatim("Delivery is not free for orders of Rs. 499 or more", source)
    assert not grounding.near_verbatim("Delivery is free for orders of Rs. 299 or more", source)


def test_coverage_check_parsing_and_switch(settings):
    from engine import coverage

    assert coverage.parse('```json\n{"covered": false}\n```') is False
    assert coverage.parse('{"covered": "true"}') is True
    assert not coverage.verdict_enabled(settings)  # offline + auto
    assert coverage.verdict_enabled(settings.with_overrides(llm_provider="openai"))
    assert not coverage.verdict_enabled(settings.with_overrides(llm_provider="openai", coverage_verdict="off"))
    v = asyncio.run(coverage.llm_covered(FakeClient('{"covered": false}'), "Do you sell gift cards?", PASSAGES))
    assert v == {"ran": True, "covered": False}
    assert asyncio.run(coverage.llm_covered(FakeClient(TimeoutError()), "q", PASSAGES))["ran"] is False


def test_coverage_check_decides_match_found_in_the_live_pipeline(settings):
    """With the check on, each search records its verdict and match_found follows it (offline stand-in: covered)."""
    from channels.dispatcher import Dispatcher
    from engine.orchestrator import SupportEngine
    from engine.state import TicketInput

    s = settings.with_overrides(coverage_verdict="llm")
    engine = SupportEngine(s, dispatcher=Dispatcher(s, dry_run=True))
    result = asyncio.run(engine.process(TicketInput(complaint="Where is my order? It was supposed to arrive yesterday.",
                                                    customer_name="Priya", order_id="ORD-4512")))
    searches = result.state.kb_results
    assert searches and all(r["coverage_verdict"]["ran"] for r in searches)
    assert all(r["match_found"] == r["coverage_verdict"]["covered"] for r in searches)
    assert result.state.ticket_id in result.reply


def test_desk_process_sentences_are_not_policy_claims():
    assert grounding.is_desk_process("A support specialist will follow up with you shortly.")
    assert grounding.is_desk_process("If it is still late, please let us know and we can check the status for you.")
    assert not grounding.is_desk_process("Contact support to get a full refund")  # a promise, still checked
    assert not grounding.is_desk_process("We will refund Rs 500 within 2 days.")
    assert not grounding.is_desk_process("They will repair or replace your laptop for free")


def test_secret_scan_covers_only_files_that_are_pushed():
    from evaluation.evaluate import pushable_files

    files = pushable_files()
    assert any(f.name == "README.md" for f in files)
    assert not any(".venv" in f.parts or f.name == ".env" for f in files)
