"""Deterministic sentiment, risk and language signals for the Customer Query Agent.

The LLM makes the final call, but it receives these signals from the
``analyze_sentiment`` tool so its triage is anchored to something measurable
(VADER polarity + anger cues + risk keywords).
"""
from __future__ import annotations

import re
from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from rag import multilingual

_ANALYZER = SentimentIntensityAnalyzer()

ANGER_CUES = [
    "ridiculous", "worst", "useless", "pathetic", "terrible", "horrible", "unacceptable", "disgusting",
    "furious", "angry", "fed up", "frustrated", "fix this now", "right now", "immediately", "asap",
    "nobody replies", "no one replies", "nobody responds", "cheated", "scam", "fraud", "never again",
    "waste of", "shame", "worst service", "how dare", "lawyer", "consumer court", "sue you",
]

RISK_PATTERNS = {
    "legal_threat": r"\b(lawyer|advocate|legal action|legal notice|sue|suing|court|consumer forum|police)\b",
    "fraud": r"\b(fraud|scam|unauthori[sz]ed|didn'?t (place|make)|never (placed|made|ordered)|hacked|"
             r"stolen|someone (else )?(logged|placed|used)|account takeover)\b",
    "payment_dispute": r"\b(charged (me )?(twice|two times|three times|thrice|multiple times|again)|"
                       r"double charge|duplicate charge|charged \w+ times|chargeback|contact(ing)? my bank|"
                       r"dispute|extra charge|overcharged)\b",
    "safety": r"\b(fire|smoke|sparks?|electric shock|shocked|burn(t|ed)?|injur(y|ed)|explod(e|ed))\b",
}

URGENT_CUES = r"\b(urgent|asap|immediately|right now|today itself|emergency|now or)\b"
UNANSWERED = r"\b(nobody|no one|no-one|none of you)\b.{0,20}\b(repl|respond|answer|call)|\bno (reply|response)\b"


def _detect_language(text: str) -> str:
    return multilingual.detect_language(text)  # script first (Tamil, Hindi, ...), langdetect for Latin text


def analyze(text: str) -> Dict:
    text = text or ""
    lower = text.lower()
    compound = _ANALYZER.polarity_scores(text)["compound"]

    cues: List[str] = [c for c in ANGER_CUES if c in lower]
    exclaims = text.count("!")
    caps_words = [w for w in re.findall(r"\b[A-Z]{3,}\b(?![-_]?\d)", text)
                  if w not in {"OTP", "UPI", "COD", "GST", "EMI", "SMS", "IST", "USB", "LED", "TV", "AC"}]
    if exclaims >= 1:
        cues.append(f"{exclaims} exclamation mark(s)")
    if caps_words:
        cues.append("shouting (CAPS)")

    risk_flags = [name for name, pat in RISK_PATTERNS.items() if re.search(pat, lower)]
    indic = multilingual.extra_signals(text)  # VADER only reads English: add Tamil / Hindi cues
    cues += [f"'{c}'" for c in indic["negative_cues"]]
    risk_flags += [f for f in indic["risk_flags"] if f not in risk_flags]
    if indic["negative_cues"] and compound >= 0:
        compound = -0.4
    anger_points = len([c for c in cues if "exclamation" not in c]) + min(exclaims, 2) * 0.5 + (1 if caps_words else 0)
    if risk_flags:
        anger_points += 0.5 * len(risk_flags)

    score = max(-1.0, min(1.0, compound - 0.15 * anger_points))
    if anger_points >= 2 or (score <= -0.5 and anger_points >= 1):
        label = "Angry"
    elif score <= -0.2 or anger_points >= 1:
        label = "Negative"
    elif score >= 0.35:
        label = "Positive"
    else:
        label = "Neutral"

    overdue = [int(n) for n in re.findall(r"\b(\d{1,3})\s*(?:days?)\s*(?:late|delayed|overdue)", lower)]
    escalating = bool(re.search(UNANSWERED, lower)) and (max(overdue, default=0) >= 7)
    if escalating and label == "Negative":
        label = "Angry"  # long overdue + ignored earlier contact = an angry customer
        score = min(score, -0.55)
    if risk_flags or escalating or (label == "Angry" and re.search(URGENT_CUES, lower)):
        urgency_hint = "High"
    elif re.search(URGENT_CUES, lower) or label in {"Angry", "Negative"}:
        urgency_hint = "Medium"
    else:
        urgency_hint = "Low/Medium (judge by business impact)"

    return {
        "sentiment": label,
        "sentiment_score": round(score, 2),
        "vader_compound": round(compound, 3),
        "anger_cues": cues,
        "risk_flags": risk_flags,
        "urgency_hint": urgency_hint,
        "overdue_days": max(overdue, default=0),
        "earlier_contact_unanswered": bool(re.search(UNANSWERED, lower)),
        "language": _detect_language(text),
    }
