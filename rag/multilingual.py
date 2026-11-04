"""Multilingual support for Indian-language complaints (Tamil, Hindi, ...).

Customers write in their own language, but the company documents are in English. Three things make
retrieval and replies work anyway:

1. **Script-aware language detection** - Tamil / Devanagari / other Indic scripts are detected from the
   characters themselves (reliable even for one-line messages), langdetect is only the fallback.
2. **English search queries** - with a real LLM the Knowledge Base Agent writes an English query (the
   model translates), and the Query Agent's English ``summary`` is searched as well. Without an LLM (the
   offline demo) a small support-domain glossary turns the key words into English. The original text is
   always searched too, so a multilingual embedding model (``EMBEDDING_BACKEND=openai`` or
   ``sentence-transformers`` with ``paraphrase-multilingual-MiniLM-L12-v2``) can match it directly.
3. **Reply language** - the detected language is passed to the agents, which answer in that language.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

LANGUAGE_NAMES = {
    "en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "ur": "Urdu",
}

# Unicode blocks -> language (Devanagari is shared by Hindi and Marathi; Hindi is the default)
SCRIPTS: List[Tuple[str, str, str]] = [
    ("ta", "஀", "௿"), ("hi", "ऀ", "ॿ"), ("te", "ఀ", "౿"),
    ("kn", "ಀ", "೿"), ("ml", "ഀ", "ൿ"), ("bn", "ঀ", "৿"),
    ("gu", "઀", "૿"), ("pa", "਀", "੿"), ("or", "଀", "୿"),
    ("ur", "؀", "ۿ"),
]

# Support-domain glossary for the offline demo (substring match: Tamil and Hindi attach suffixes to words).
# Only used when no LLM is available to translate; it covers the vocabulary of the company documents.
GLOSSARY: Dict[str, str] = {
    # ---- Tamil
    "ஆர்டர்": "order", "டெலிவரி": "delivery", "வரவில்லை": "not arrived", "வந்து சேரவில்லை": "not arrived",
    "தாமத": "late delayed", "பார்சல்": "package", "ரீஃபண்ட்": "refund", "ரிஃபண்ட்": "refund",
    "பணம் திரும்ப": "refund", "பணத்தை திரும்ப": "refund", "திருப்பி": "return", "திரும்ப": "return",
    "உடைந்": "broken damaged", "உடைஞ்": "broken damaged", "சேதம": "damaged", "பழுது": "defective repair",
    "வேலை செய்யவில்லை": "not working defective", "மாற்ற": "replacement exchange", "மாற்றி": "replacement exchange",
    "இரண்டு முறை": "charged twice", "கட்டணம்": "charge fee payment", "பணம்": "money payment",
    "கணக்கு": "account", "கடவுச்சொல்": "password", "உள்நுழை": "log in", "ஓடிபி": "otp",
    "பால்": "milk", "கெட்டு": "spoiled", "காய்கறி": "vegetables", "மளிகை": "grocery",
    "நிறுவல்": "installation", "இன்ஸ்டாலேஷன்": "installation", "டெக்னீஷியன்": "technician",
    "பேட்டரி": "battery", "வீங்க": "swelling", "சூடா": "hot overheating", "போன்": "phone", "கைபேசி": "phone",
    "டிவி": "tv", "மிக்ஸி": "blender mixer", "பிளெண்டர்": "blender", "ஏசி": "ac air conditioner",
    "காலணி": "shoes", "செருப்பு": "sandals shoes", "ஷூ": "shoes", "அளவு": "size", "உத்தரவாத": "warranty",
    "கூப்பன்": "promo code", "தள்ளுபடி": "discount promo", "உறுப்பினர்": "membership", "சந்தா": "subscription",
    "மோசடி": "fraud", "திருட": "stolen fraud", "புகார்": "complaint",
    # ---- Hindi
    "ऑर्डर": "order", "आर्डर": "order", "डिलीवरी": "delivery", "नहीं आया": "not arrived", "नहीं पहुंचा": "not arrived",
    "देर": "late delayed", "पार्सल": "package", "रिफंड": "refund", "पैसे वापस": "refund", "वापस": "return",
    "टूट": "broken damaged", "खराब": "defective damaged", "काम नहीं": "not working defective",
    "बदल": "replacement exchange", "दो बार": "charged twice", "चार्ज": "charge", "भुगतान": "payment",
    "पैसे": "money payment", "खाता": "account", "अकाउंट": "account", "पासवर्ड": "password", "लॉगिन": "log in",
    "ओटीपी": "otp", "दूध": "milk", "सड़ा": "spoiled", "सड़ गय": "spoiled", "फट गया": "spoiled", "सब्जी": "vegetables",
    "किराना": "grocery", "इंस्टॉलेशन": "installation", "इंस्टालेशन": "installation", "टेक्नीशियन": "technician",
    "बैटरी": "battery", "फूल गई": "swelling", "फूल गया": "swelling", "गरम": "hot overheating", "फोन": "phone", "फ़ोन": "phone",
    "टीवी": "tv", "ब्लेंडर": "blender", "मिक्सर": "blender mixer", "एसी": "ac air conditioner",
    "जूत": "shoes", "साइज": "size", "साइज़": "size", "वारंटी": "warranty", "कूपन": "promo code",
    "छूट": "discount promo", "सदस्यता": "membership", "धोखा": "fraud", "शिकायत": "complaint",
}

# Sentiment / risk cues VADER (English-only) cannot see
NEGATIVE_CUES = ["மோசம்", "கோபம்", "ஏமாற்", "வெறுப்பு", "கேவலம்", "बेकार", "गुस्सा", "बहुत बुरा", "घटिया", "परेशान"]
RISK_CUES = {
    "fraud": ["மோசடி", "திருட", "धोखा", "धोखाधड़ी", "चोरी"],
    "legal_threat": ["வழக்கு", "நீதிமன்ற", "நுகர்வோர் நீதிமன்ற", "कोर्ट", "कानूनी", "उपभोक्ता अदालत"],
    "safety": ["தீப்பிடி", "தீ பிடி", "புகை வந்", "மின்சாரம் தாக்", "आग लग", "धुआं", "करंट"],
}


def script_language(text: str) -> str:
    """Language from the writing system, or '' when the text is (mostly) Latin script."""
    letters = [ch for ch in text or "" if ch.isalpha() or "ऀ" <= ch <= "෿"]
    if not letters:
        return ""
    counts: Dict[str, int] = {}
    for ch in letters:
        for code, lo, hi in SCRIPTS:
            if lo <= ch <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return ""
    code, n = max(counts.items(), key=lambda kv: kv[1])
    return code if n / len(letters) >= 0.3 else ""


def detect_language(text: str) -> str:
    """ISO code of the customer's language: script first (Tamil, Hindi, ...), then langdetect for Latin text."""
    text = text or ""
    by_script = script_language(text)
    if by_script:
        return by_script
    letters = re.findall(r"[^\W\d_]", text)
    if not letters or sum(1 for ch in letters if ord(ch) < 128) / len(letters) > 0.9:
        return "en"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get((code or "").split("-")[0], code or "unknown")


def is_english(text: str) -> bool:
    return detect_language(text) in {"en", "unknown"}


def glossary_translate(text: str) -> str:
    """English keywords for an Indian-language message (offline fallback; the LLM translates when available)."""
    found: List[str] = []
    for word, english in sorted(GLOSSARY.items(), key=lambda kv: -len(kv[0])):  # longest phrases first
        if word in (text or "") and english not in found:
            found.append(english)
    latin = " ".join(re.findall(r"[A-Za-z][A-Za-z0-9-]+", text or ""))  # English words mixed into the message
    return " ".join(found + ([latin] if latin else [])).strip()


def search_queries(query: str, english_hint: str = "") -> List[str]:
    """Queries to run for one search: the query itself, plus English versions when it is not English."""
    queries = [q for q in [query] if q and q.strip()]
    if query and not is_english(query):
        if english_hint and is_english(english_hint):
            queries.append(english_hint)  # e.g. the Query Agent's English summary of the complaint
        translated = glossary_translate(query)
        if translated:
            queries.append(translated)
    return list(dict.fromkeys(queries))


def extra_signals(text: str) -> Dict[str, List[str]]:
    """Negative-sentiment and risk cues in Tamil / Hindi text (VADER only understands English)."""
    text = text or ""
    return {
        "negative_cues": [c for c in NEGATIVE_CUES if c in text],
        "risk_flags": [flag for flag, cues in RISK_CUES.items() if any(c in text for c in cues)],
    }
