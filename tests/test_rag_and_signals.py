import pytest

from engine import escalation
from engine.parsing import extract_json, normalize_category, normalize_resolution
from rag.knowledge import KnowledgeStore
from rag.sentiment import analyze

PDF_CASES = [
    ("Where is my order? It was supposed to arrive yesterday.", {"shipping_policy.md"}),
    ("I want to return the shoes I bought last week. How do I get a refund?", {"refund_policy.md", "fashion_footwear.md"}),
    ("I forgot my password and can't log in.", {"account_and_login.md"}),
    ("The blender I received is broken. Can I get a replacement?", {"product_faq.md", "home_appliances.md"}),
]


@pytest.mark.parametrize("query,sources", PDF_CASES)
def test_policy_rag_finds_the_right_document(settings, query, sources):
    res = KnowledgeStore(settings).search_kb(query)
    assert res["match_found"]
    assert res["policy_passages"][0]["source"] in sources


LINE_CASES = [
    ("My new phone battery is swelling and it gets very hot while charging.", "electronics", "electronics.md"),
    ("The blender I received is broken. Can I get a replacement?", "home_appliances", "home_appliances.md"),
    ("Can I exchange these sneakers for a bigger size?", "fashion", "fashion_footwear.md"),
    ("The milk and paneer in my grocery order were spoiled.", "grocery", "remya_fresh_grocery.md"),
    ("The technician did not come for my AC installation today.", "installation_repair", "remya_care_installation_repair.md"),
    ("How do I raise a claim on my Remya Shield screen damage plan?", "protection_plan", "remya_shield_protection_plans.md"),
    ("I was charged for a Remya Plus renewal I did not want.", "membership", "remya_plus_membership.md"),
]


@pytest.mark.parametrize("text,line,doc", LINE_CASES)
def test_catalog_detects_line_and_scoped_rag_uses_its_guide(settings, text, line, doc):
    store = KnowledgeStore(settings)
    assert store.catalog.identify(text)["primary"] == line
    res = store.search_kb(text, product_line=line)
    assert res["match_found"] and res["policy_passages"][0]["source"] == doc
    # scoping: nothing from another product line's guide can appear
    assert {p["scope"] for p in res["policy_passages"]} <= {line, "general"}


def test_multi_line_ticket_and_exclusions(settings):
    catalog = KnowledgeStore(settings).catalog
    multi = catalog.identify("My TV installation is pending and the shoes are the wrong size")
    assert {"installation_repair", "electronics", "fashion"} <= set(multi["all"])
    assert catalog.identify("I changed my mobile number and can't get the OTP")["primary"] == "general"
    assert catalog.identify("anything", hint="Electronics")["primary"] == "electronics"


def test_line_risks_and_routing(settings):
    catalog = KnowledgeStore(settings).catalog
    risks = catalog.identify("I found an insect in the paneer from my grocery order")["line_risks"]
    assert risks and risks[0]["flag"] == "food_safety"
    assert catalog.route("grocery", [])["name"] == "Remya Fresh Grocery Desk"
    assert catalog.route("electronics", ["fraud"])["name"] == "Trust & Safety Team"
    assert catalog.route("membership", ["payment_dispute"])["name"] == "Billing Desk"


def test_uncovered_question_is_not_matched(settings):
    res = KnowledgeStore(settings).search_kb("Do you sell gift cards for a company of 500 employees with custom branding?")
    assert not res["match_found"]


def test_synonym_expansion_helps_customer_phrasing(settings):
    res = KnowledgeStore(settings).search_kb("How can I send back the shoes and get my money refunded?")
    assert res["match_found"] and res["policy_passages"][0]["source"] == "refund_policy.md"


def test_learned_rag_adds_and_deduplicates(settings):
    store = KnowledgeStore(settings)
    first = store.add_learned_solution(ticket_id="TKT-0001", category="Refund", question="return shoes refund",
                                       answer="Use My Orders > Return.", sources="refund_policy.md", confidence=90)
    second = store.add_learned_solution(ticket_id="TKT-0002", category="Refund", question="return shoes refund",
                                        answer="Use My Orders > Return.", sources="refund_policy.md", confidence=90)
    assert first["action"] == "added" and second["action"] == "updated"
    assert len(store.learned_entries()) == 1
    assert store.search_kb("how do I return my shoes for a refund")["resolved_ticket_matches"]


def test_sentiment_flags_angry_legal_payment():
    a = analyze("You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.")
    assert a["sentiment"] == "Angry"
    assert {"legal_threat", "payment_dispute"} <= set(a["risk_flags"])
    assert a["urgency_hint"] == "High"


def test_sentiment_neutral_question():
    a = analyze("I forgot my password and can't log in.")
    assert a["sentiment"] in {"Neutral", "Negative"} and not a["risk_flags"]


def test_order_number_is_not_shouting():
    assert "shouting (CAPS)" not in analyze("Where is ORD-4512?")["anger_cues"]


def test_escalation_rules(settings):
    calm = {"urgency": "Low", "sentiment": "Neutral", "sentiment_score": 0.0, "risk_flags": []}
    assert escalation.decide(calm, {"match_found": True}, {"confidence": 92}, settings)["status"] == "Resolved"
    assert "No matching knowledge-base document" in escalation.decide(calm, {"match_found": False}, {"confidence": 92},
                                                                      settings)["reasons"]
    low = escalation.decide(calm, {"match_found": True}, {"confidence": 40}, settings)
    assert low["status"] == "Escalated" and any("confidence" in r for r in low["reasons"])
    risky = escalation.apply_guardrails({**calm, "risk_flags": ["fraud"]})
    assert risky["urgency"] == "High"
    assert escalation.decide(calm, {"match_found": True}, {"confidence": 95}, settings, agent_escalate=True,
                             agent_reason="customer asked for a manager")["status"] == "Escalated"


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('Here you go: {"a": {"b": 2}} thanks')["a"]["b"] == 2
    assert extract_json("{'a': 1,}") == {"a": 1}
    assert extract_json("no json") is None


def test_normalisers():
    assert normalize_category("Delivery issue") == "Shipping"
    assert normalize_category("payments") == "Billing"
    assert normalize_resolution({"response": "x", "confidence": 0.9}, "")["confidence"] == 90
    assert normalize_resolution(None, "plain reply, confidence: 70")["confidence"] == 70


# ---------------------------------------------------------------- buildathon feedback: multilingual, rerank, grounding
def test_indic_tokens_survive_and_language_is_detected():
    from rag.multilingual import detect_language, glossary_translate, search_queries
    from rag.text import tokenize

    assert "டெலிவரி" in tokenize("என் ஆர்டர் டெலிவரி ஆகவில்லை") and "ब्लेंडर" in tokenize("मेरा ब्लेंडर टूटा है")
    assert detect_language("என் ஆர்டர் வரவில்லை") == "ta" and detect_language("मेरा ऑर्डर नहीं आया") == "hi"
    assert detect_language("My order is late") == "en"
    assert "refund" in glossary_translate("பணம் திரும்ப வேண்டும்")
    assert search_queries("My order is late") == ["My order is late"]  # English: no extra queries
    assert len(search_queries("मेरा ऑर्डर नहीं आया", "my order has not arrived")) == 3


def test_tamil_and_hindi_questions_reach_the_right_policy(settings):
    store = KnowledgeStore(settings)
    ta = store.search_kb("நான் வாங்கிய மிக்ஸி உடைந்து வந்தது, மாற்றித் தர முடியுமா?", product_line="home_appliances")
    hi = store.search_kb("एक ही ऑर्डर के लिए दो बार पैसे कट गए", product_line="general")
    assert ta["query_language"] == "ta" and ta["policy_passages"][0]["source"] == "home_appliances.md"
    assert hi["policy_passages"][0]["source"] == "billing_and_payments.md" and "charged twice" in \
        hi["policy_passages"][0]["section"]
    # without the translation step the Tamil question finds nothing useful
    assert not store.search_kb("நான் வாங்கிய மிக்ஸி உடைந்து வந்தது", product_line="home_appliances",
                               translate=False)["match_found"]


def test_every_passage_explains_how_it_was_found(settings):
    res = KnowledgeStore(settings).search_kb("My blender arrived broken", product_line="home_appliances")
    top = res["policy_passages"][0]
    assert top["found_by"] in {"bm25+vector", "bm25", "vector"} and top["ranks"]["fusion"] >= 1
    assert top["rerank_score"] is not None and "reranker: feature" in res["retriever"]
    assert top["chunk_id"]


def test_grounding_flags_invented_figures_and_bad_citations():
    from engine import grounding

    refs = {"S1": {"source": "refund_policy.md", "section": "Refund timelines", "chunk_id": "x"}}
    passages = [{"text": "Refunds reach the original payment method within 5 to 7 business days."}]
    ok = grounding.check("Hi Ana,\n\nYour refund reaches you within 5 to 7 business days.", ["S1"], refs, passages,
                         ids=["TKT-0001"])
    assert ok["grounded"] and ok["valid_citations"] == ["S1"] and ok["supported_ratio"] == 1.0
    bad = grounding.check("Your refund of Rs 2000 arrives within 3 days and you get a 10% voucher.", ["S1", "S9"],
                          refs, passages)
    assert not bad["grounded"] and bad["invalid_citations"] == ["S9"]
    assert set(bad["unsupported_figures"]) == {"2000", "3", "10"}


def test_ungrounded_reply_is_escalated(settings):
    from engine.escalation import evaluate_rules

    reasons = evaluate_rules({"urgency": "Low"}, {"match_found": True}, {"confidence": 95}, settings,
                             grounding={"unsupported_figures": ["499"]})
    assert any("figures not found" in r for r in reasons)


def test_index_is_saved_and_reloaded_until_a_document_changes(settings, tmp_path):
    import shutil

    from rag import knowledge

    kb = tmp_path / "kb"
    shutil.copytree(settings.kb_dir, kb)
    s = settings.with_overrides(kb_dir=kb)
    knowledge._POLICY_CACHE.clear()
    first = KnowledgeStore(s).index_info()
    assert first["origin"] == "built" and (s.data_dir / "index" / "policy" / "manifest.json").exists()
    knowledge._POLICY_CACHE.clear()
    assert KnowledgeStore(s).index_info()["origin"] == "loaded from disk"  # nothing changed: no re-embedding
    doc = kb / "general" / "refund_policy.md"
    doc.write_text(doc.read_text(encoding="utf-8") + "\n\n## Gift wrap\nGift wrap is free.\n", encoding="utf-8")
    knowledge._POLICY_CACHE.clear()
    info = KnowledgeStore(s).index_info()
    assert info["origin"] == "built" and info["chunks"] == first["chunks"] + 1
    knowledge._POLICY_CACHE.clear()
