"""Escalation rule engine (assignment Task 5).

The Log Agent gives its own judgement (LLM), and this deterministic layer
guarantees the mandatory rules are never missed. Final decision = rules OR agent.
"""
from __future__ import annotations

from typing import Dict, List

from engine.config import Settings

RISK_LABELS = {
    "legal_threat": "legal threat",
    "fraud": "fraud / unauthorised activity",
    "payment_dispute": "payment dispute",
    "safety": "safety hazard",
    # product-line / service specific risks (catalog.json -> risk_keywords)
    "product_safety": "product safety issue",
    "food_safety": "food safety concern",
    "service_misconduct": "technician conduct or property damage",
    "claim_dispute": "rejected protection-plan claim",
    "prompt_injection": "attempt to manipulate the AI",
}
# risks that also force High urgency (the others escalate but keep the agent's urgency)
HIGH_URGENCY_RISKS = {"legal_threat", "fraud", "payment_dispute", "safety", "product_safety", "food_safety"}


def evaluate_rules(triage: Dict, kb: Dict, resolution: Dict, settings: Settings,
                   manager: Dict | None = None, grounding: Dict | None = None) -> List[str]:
    reasons: List[str] = []
    if manager and manager.get("needs_human"):
        reasons.append("Support Manager could not match a division - needs a human")
    if triage.get("urgency") == "High":
        reasons.append("High urgency")
    if triage.get("sentiment") == "Angry" and float(triage.get("sentiment_score", 0)) <= settings.very_angry_score:
        reasons.append("Very angry customer")
    for flag in triage.get("risk_flags", []):
        if flag in RISK_LABELS:
            reasons.append(f"Involves {RISK_LABELS[flag]}")
    if not kb.get("match_found", False):
        reasons.append("No matching knowledge-base document")
    conf = resolution.get("confidence")
    if conf is not None and int(conf) < settings.confidence_threshold:
        reasons.append(f"Low resolution confidence ({conf}% < {settings.confidence_threshold}%)")
    if grounding and grounding.get("unsupported_figures"):  # post-generation check (engine/grounding.py)
        reasons.append("Reply states figures not found in the documents ("
                       + ", ".join(grounding["unsupported_figures"][:5]) + ")")
    if grounding and grounding.get("unsupported_claims"):  # LLM claim verifier
        reasons.append(f"Reply makes {len(grounding['unsupported_claims'])} claim(s) the documents do not support")
    return reasons


def decide(triage: Dict, kb: Dict, resolution: Dict, settings: Settings, agent_escalate: bool = False,
           agent_reason: str = "", manager: Dict | None = None, grounding: Dict | None = None) -> Dict:
    rule_reasons = evaluate_rules(triage, kb, resolution, settings, manager, grounding)
    reasons = list(rule_reasons)
    agent_reason = (agent_reason or "").strip()
    if agent_escalate and agent_reason:
        if not any(agent_reason.lower() in r.lower() or r.lower() in agent_reason.lower() for r in reasons):
            reasons.append(f"Log Agent: {agent_reason}")
    elif agent_escalate:
        reasons.append("Log Agent judged the case complex")
    escalate = bool(reasons)
    return {
        "escalate": escalate,
        "status": "Escalated" if escalate else "Resolved",
        "reasons": reasons,
        "rule_reasons": rule_reasons,
        "agent_escalate": bool(agent_escalate),
        "agent_reason": agent_reason,
        "agreement": bool(rule_reasons) == bool(agent_escalate),
    }


def apply_guardrails(triage: Dict) -> Dict:
    """Business guardrail: fraud, payment disputes, legal threats, safety and food-safety issues are always High."""
    if any(f in HIGH_URGENCY_RISKS for f in triage.get("risk_flags", [])) and triage.get("urgency") != "High":
        triage = dict(triage)
        triage["urgency_llm"] = triage.get("urgency")
        triage["urgency"] = "High"
        triage["urgency_reason"] = (triage.get("urgency_reason", "") + " [guardrail: risk flag forces High]").strip()
    return triage


def compute_priority(triage: Dict) -> int:
    """0-100 priority score used to sort the queue and the dashboard."""
    base = {"High": 70, "Medium": 40, "Low": 15}.get(triage.get("urgency"), 40)
    sentiment = {"Angry": 15, "Negative": 8}.get(triage.get("sentiment"), 0)
    risk = 5 * len([f for f in triage.get("risk_flags", []) if f in RISK_LABELS])
    repeat = 10 if triage.get("repeat_issue") else 0
    return int(min(100, base + sentiment + risk + repeat))
