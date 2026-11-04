"""Small text utilities shared by the retrievers (tokenising, stemming, chunking)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "been",
    "before", "being", "but", "by", "can", "could", "did", "do", "does", "doing", "done", "for", "from", "get",
    "got", "had", "has", "have", "having", "he", "her", "here", "him", "his", "how", "i", "i'm", "if", "in", "into",
    "is", "it", "it's", "its", "just", "me", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "out", "own", "please", "same", "she", "should", "so", "some", "still", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "why", "will", "with", "would", "you", "your", "yours", "hi", "hello", "dear", "team", "thanks", "thank",
    "can't", "cannot", "dont", "don't", "im", "ive", "i've", "want", "need", "know", "let", "via", "one", "ok",
    "even", "though", "although", "despite", "also", "already", "yet", "any", "anything",
}
# Conversational filler that customers use but policies never contain ("last week", "nobody replies").
# Removed from the coverage check only, so they do not make a covered question look uncovered.
FILLER = {
    "yesterday", "today", "tomorrow", "week", "weeks", "last", "ago", "day", "days", "month", "months", "time",
    "times", "bought", "buy", "purchased", "got", "nobody", "anyone", "someone", "replies", "reply", "respond",
    "responds", "emails", "mail", "again", "already", "really", "please", "kindly", "sir", "madam", "help",
    "said", "says", "tell", "told", "asap", "urgent", "fix", "happened", "issue", "problem", "suppose", "supposed",
    "found", "disgusting", "terrible", "horrible", "worst", "pathetic", "ridiculous", "unacceptable", "morning",
    "evening", "tonight", "whole",
}
_FILLER_STEMS = set(_STEMMER.stemWords(list(FILLER)))


# Customer phrasing -> the words our policies use (applied to search queries only).
DOMAIN_SYNONYMS = [
    (r"\b(send|give|ship) (it |them |this |these |the \w+ )?back\b", "return"),
    (r"\b(money back|my money|refunded|reimburse(ment|d)?)\b", "refund"),
    (r"\bparcel\b", "package"),
    (r"\b(hasn'?t|has not|not yet|never|haven'?t|didn'?t|did not)\s+(arrived?|received?|come|got|delivered)\b",
     "not arrived late"),
    (r"\b(sign[- ]?in|log on|logon)\b", "log in"),
    (r"\b(hacked|compromised)\b", "someone else logged in account"),
    (r"\b(double[- ]charged|charged twice|charged two times|charged \w+ times)\b", "charged twice more than once"),
    (r"\b(coupon|voucher|discount code)\b", "promo code"),
    (r"\b(sneakers?|trainers?|loafers?|heels|flip[- ]flops?)\b", "shoes"),
    (r"\b(smells? (sour|bad|off|weird|funny)|tastes? (sour|bad|off)|stinks?|rancid|curdled|mouldy|moldy)\b",
     "spoiled"),
    (r"\bpacket (was |is )?(leaking|torn|burst)|(leaking|torn|burst) packet\b", "damaged packaging"),
    (r"\b(insects?|worms?|cockroach(es)?|hair|glass pieces?|stones?|plastic pieces?)\b", "foreign object"),
    (r"\b(too (tight|small|big|large|loose)|(bigger|smaller|larger) size|one size (up|down|bigger|smaller|larger)"
     r"|wrong size)\b", "different size"),
    (r"\b(cracked|broken|shattered) (screen|display)\b|\b(screen|display) (cracked|broke|broken|shattered)\b",
     "screen damage"),
    (r"\b(faulty|stopped working|dead on arrival)\b", "defective not working"),
    # devices and appliances
    (r"\b(won'?t|will not|doesn'?t|does not|not) (turn|switch|power) on\b|\bno power\b", "dead does not switch on"),
    (r"\b(bulging|bloated|puffed up|swollen)\b", "swelling"),
    (r"\b(not cooling|no cooling|not cold|not heating|not spinning|not draining)\b", "not working"),
    (r"\b(fridge)\b", "refrigerator"),
    (r"\b(mixie|mixer grinder|grinder|juicer)\b", "mixer blender small appliance"),
    (r"\b(burning smell|smell of burning|smoke came|caught fire)\b", "smelled burning smoke"),
    # account and payments
    (r"\b(verification code|one[- ]time password|security code)\b", "otp"),
    (r"\b(remove|close|deactivate|erase) (my )?account\b", "delete account"),
    (r"\b(can'?t|cannot|unable to) (get into|access|open) (my )?account\b", "cannot log in account"),
    (r"\b(debited|deducted|cut from my (bank|account))\b", "money deducted"),
    (r"\b(billed|charged) (two|2|three|3) times\b", "charged twice"),
    (r"\b(gst bill|tax invoice|bill copy)\b", "gst invoice"),
    (r"\b(pay (in )?cash|cash payment)\b", "cash on delivery cod payment methods"),
    # orders and delivery
    (r"\b(ripped|torn|tampered|seal broken)\b", "torn open damaged"),
    (r"\b(not inside|was not in|wasn'?t in|missing from)\b", "missing"),
    (r"\b(still hasn'?t shown up|hasn'?t shown up|not shown up|never showed up|didn'?t show up|no[- ]show)\b",
     "did not come not arrived late"),
    (r"\b(send|sent|deliver|delivered|ship|shipped) (it |the order |my order )?to (my |a )?"
     r"(office|another|different|new|other)( address| place| location)?\b"
     r"|\b(change|update) (the |my )?(shipping|delivery) address\b", "change delivery address"),
    (r"\binstead of\b(?=[^.?!]{0,50}\b(i|we) (had )?ordered\b)", "different item than ordered instead of"),
    (r"\b(original|authentic|counterfeit|fake|duplicate product|first copy)\b", "genuine"),
    # support
    (r"\b(how (fast|soon|quickly)|how long) (will|does|do) (someone|you|support|your team) "
     r"(reply|respond|answer|get back)\b", "how quickly will support respond response time"),
    (r"\b(working hours|office hours|timings|open hours)\b", "hours contact support"),
]


def expand_query(text: str) -> str:
    out = text or ""
    for pattern, canonical in DOMAIN_SYNONYMS:
        out = re.sub(pattern, canonical, out, flags=re.I)
    return out


# Latin words (with "don't"-style suffixes) or runs of Indic script. Indic vowel signs and the virama are
# combining marks that Python's \w does not match, so the whole Unicode block is used: Tamil "வணக்கம்"
# stays one token instead of being cut into letters. The dandas (U+0964/0965) are punctuation, not letters.
TOKEN_RX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[\u0900-\u0963\u0966-\u0DFF]+")
INDIC_STOPWORDS = {
    # Tamil
    "என்", "எனது", "என்னுடைய", "நான்", "ஒரு", "இது", "அது", "மற்றும்", "இந்த", "அந்த", "உள்ளது", "வேண்டும்",
    # Hindi
    "है", "हैं", "था", "थी", "का", "की", "के", "को", "में", "से", "पर", "और", "मेरा", "मेरी", "मेरे", "मैं", "यह",
    "वह", "एक", "भी", "तो", "हो", "गया", "गई", "कर", "क्या",
}


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens (English stemmed; Tamil, Hindi and other Indic words kept whole), stop-words removed."""
    words = TOKEN_RX.findall((text or "").lower())
    words = [w for w in words if w not in STOPWORDS and w not in INDIC_STOPWORDS and (len(w) > 1 or not w.isascii())]
    return [_STEMMER.stemWord(w) if w.isascii() else w for w in words]


def content_terms(tokens: List[str]) -> List[str]:
    """Stemmed tokens minus filler words and bare numbers (used for coverage scoring)."""
    return [t for t in tokens if t not in _FILLER_STEMS and not t.isdigit()]


_ABBREV = re.compile(r"\b(Rs|No|e\.g|i\.e|approx|Mr|Ms|Dr|vs)\.", flags=re.I)


def split_sentences(text: str) -> List[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\u2024"), text or "")  # keep "Rs. 450" together
    parts = re.split(r"(?<=[.!?])\s+|\n+", protected)
    parts = [p.replace("\u2024", ".").strip(" -*\t") for p in parts]
    return [p for p in parts if len(p) > 3]


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    text: str
    collection: str = "policy"
    meta: Dict = field(default_factory=dict)

    @property
    def searchable(self) -> str:
        return self.meta.get("search_text") or f"{self.section}\n{self.text}"


CHUNKER_VERSION = "markdown-sections-v2"  # stored in the index manifest: a new chunker forces a rebuild


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunk_markdown(path: Path, scope: str = "general") -> List[Chunk]:
    """Split a markdown policy file into section-aware retrieval chunks.

    Each `###` question (or `##` section without sub-questions) becomes one chunk, prefixed with its parent
    heading, so a chunk always holds ONE complete answer (eligibility, steps and timelines stay together)
    instead of an arbitrary 500-character window. Every chunk carries metadata: a stable id (SHA-1 of the
    document, section and text - identical on every run, so re-indexing never duplicates chunks), the
    document title, its SHA-256 and last-updated date, the division (scope) and the document type.
    """
    raw = path.read_text(encoding="utf-8")
    source = path.name
    doc_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    updated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    title_match = re.search(r"^# (.+)$", raw, flags=re.M)
    doc_title = title_match.group(1).strip() if title_match else source
    chunks: List[Chunk] = []
    h2 = ""
    current_title = ""
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body and current_title:
            section = f"{h2} > {current_title}" if h2 and current_title != h2 else current_title
            chunk_id = hashlib.sha1(f"{source}|{section}|{body}".encode("utf-8")).hexdigest()[:12]
            chunks.append(
                Chunk(id=f"{source}#{len(chunks) + 1}", source=source, section=section, text=body, collection="policy",
                      meta={"scope": scope, "chunk_id": chunk_id, "doc_title": doc_title, "doc_type": "policy",
                            "doc_sha256": doc_hash[:16], "updated": updated, "position": len(chunks) + 1})
            )

    for line in raw.splitlines():
        if line.startswith("### "):
            flush()
            buffer = []
            current_title = line[4:].strip()
        elif line.startswith("## "):
            flush()
            buffer = []
            h2 = line[3:].strip()
            current_title = h2
        elif line.startswith("# "):
            continue
        else:
            buffer.append(line)
    flush()
    return chunks
