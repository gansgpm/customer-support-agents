"""Guardrails for the ticketing tool.

Input guardrail  (``check_input``)  - runs before any LLM call, costs nothing:
    * blocks messages that are not support requests: empty, greetings / thanks / "test", gibberish,
      spam or promotions, pure abuse, prompt-injection attempts, duplicates and flooding;
    * masks sensitive data (card numbers, CVV, OTP, passwords, 12-digit ID numbers) before the text
      reaches the agents, the log or the knowledge base;
    * caps very long messages.
    A real complaint that merely *contains* bad language or an injection attempt is NOT blocked:
    it goes through with a flag (an injection attempt always escalates to a human).

Manager guardrail - the Support Manager can answer ``ROUTE: none`` for off-topic or other
non-support messages the rules above miss; the desk then stops before any division group runs.

Reply guardrail  (``sanitize_reply``) - strips AI self-talk, internal agent notes, JSON / markdown
debris and over-long text from the reply before the customer sees it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

MAX_CHARS = 3000

GREETINGS = {
    "hi", "hii", "hiii", "hai", "hello", "helo", "hey", "heyy", "yo", "ok", "okay", "okk", "k", "kk", "thanks",
    "thank", "you", "thankyou", "thx", "ty", "good", "morning", "afternoon", "evening", "night", "gm", "gn", "bye",
    "goodbye", "see", "later", "test", "testing", "tested", "yes", "no", "yeah", "yep", "nope", "fine", "cool",
    "nice", "great", "lol", "haha", "hmm", "hmmm", "sir", "madam", "mam", "bro", "dear", "team", "there", "all",
    "welcome", "noted", "done", "sure", "alright", "awesome", "super", "very", "much", "so", "a", "lot",
    "is", "this", "working", "anyone", "vanakkam", "namaste", "please", "pls", "plz",
}
# words that show the message is about the company's business (keep these messages)
SUPPORT_CUES = re.compile(
    r"\b(order|orders|deliver\w*|arriv\w*|ship\w*|track\w*|courier|parcel|package|refund\w*|return\w*|exchang\w*|"
    r"cancel\w*|replac\w*|broken|damag\w*|defect\w*|not working|stopped|missing|wrong|late|delay\w*|charg\w*|"
    r"pay\w*|paid|debit\w*|card|upi|wallet|invoice|bill\w*|promo|coupon|discount|price|offer|account|login|log in|"
    r"password|otp|sign|install\w*|technician|repair\w*|warranty|claim|plan|membership|subscri\w*|renew\w*|"
    r"product|item|size|grocery|milk|food|phone|laptop|tv|blender|appliance|shoe\w*|sell|buy|purchas\w*|gift|"
    r"bulk|store|remya industries|remya ?fresh|remya ?care|remya ?shield|remya ?plus|ticket|complain\w*|issue|problem|help|support|"
    r"ord[-_ ]?\d+)\b", re.I)
SPAM_PATTERNS = re.compile(
    r"(earn (money|\$|rs|₹)|make money|work from home|crypto|bitcoin|forex|trading signal|investment plan|"
    r"lottery|you (have )?won|winner|prize|jackpot|casino|betting|click here|limited offer|free money|"
    r"loan (offer|approved)|seo services|buy followers|increase followers|dm me|whatsapp me on|"
    r"adult|xxx|viagra|telegram channel|join (my|our) (channel|group)|promo code for you)", re.I)
URL = re.compile(r"(https?://|www\.)\S+", re.I)
INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)|"
    r"disregard (all |the )?(previous|above|your) (instructions|rules)|forget (your|all) (instructions|rules)|"
    r"(reveal|show|print|tell me) (your |the )?(system )?(prompt|instructions)|system prompt|"
    r"you are now|act as (a|an|my)|pretend (to be|you are)|jailbreak|developer mode|\bDAN\b|"
    r"override (the )?(policy|rules)|new instructions:|### ?instruction)", re.I)
ABUSE = re.compile(
    r"\b(idiot\w*|stupid|fool\w*|moron\w*|bastard\w*|bloody|damn|shit\w*|fuck\w*|f\*+k|bitch\w*|"
    r"asshole\w*|scoundrel\w*|nonsense people|useless (people|fellows|idiots)|go to hell|shut up)\b", re.I)

PII_RULES = [
    (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), lambda m: f"[card ending {re.sub(r'\D', '', m.group(0))[-4:]}]", "card_number"),
    (re.compile(r"\b(cvv|cvc)\s*(no\.?|number|is|:)?\s*\d{3,4}\b", re.I), lambda m: "[CVV removed]", "cvv"),
    (re.compile(r"\b(otp|one[- ]time password|verification code)\s*(is|was|:|-)?\s*\d{4,8}\b", re.I),
     lambda m: "[OTP removed]", "otp"),
    (re.compile(r"\b(password|passcode|pin)\s*(is|was|:|-)\s*\S+", re.I), lambda m: "[password removed]", "password"),
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), lambda m: "[ID number removed]", "id_number"),
]

REPLIES = {
    "empty": None,
    "chit_chat": ("Hi{name}! I'm {company}'s support assistant. Please tell me what went wrong - for example your "
                  "order number and the problem with your order, product or service - and I'll help right away."),
    "gibberish": ("Hi{name}, I couldn't understand that message. Please describe your issue in a sentence or two, "
                  "with your order number if you have one."),
    "abuse": ("Hi{name}, we want to help, but we can only respond to respectful messages. If you have a problem with "
              "an order, product or service, please describe it and include your order number."),
    "prompt_injection": ("Hi{name}, I can only help with {company} orders, products and services. Please describe "
                         "the problem you need help with."),
    "off_topic": ("Hi{name}, I can only help with {company} orders, products and services. If you have a question "
                  "about one of those, please describe it and I'll be glad to help."),
    "spam": None,  # never answer spam
    "duplicate": ("Hi{name}, we have already received this message as ticket {previous}. Our team is working on it - "
                  "there's no need to send it again."),
    "flooding": ("Hi{name}, we've received several messages from you in a short time. They are all being handled; "
                 "please wait for our reply on ticket {previous}."),
}
LABELS = {
    "empty": "empty message", "chit_chat": "greeting / chit-chat, no support request", "gibberish": "unreadable text",
    "spam": "spam or promotion", "abuse": "abusive message with no support request",
    "prompt_injection": "attempt to manipulate the AI", "off_topic": "not about Remya Industries Ltd orders, products or services",
    "duplicate": "duplicate of an earlier ticket", "flooding": "too many messages in a short time",
}


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""  # block reason key (see LABELS)
    label: str = ""
    flags: List[str] = field(default_factory=list)  # allowed-with-flags: pii_masked:card_number, abusive_language...
    clean_text: str = ""
    reply: Optional[str] = None
    previous_ticket: str = ""


# --------------------------------------------------------------------------- helpers
def mask_pii(text: str) -> tuple[str, List[str]]:
    found: List[str] = []
    for pattern, repl, kind in PII_RULES:
        text, n = pattern.subn(repl, text)
        if n:
            found.append(kind)
    return text, found


def _words(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def _is_latin(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text)
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) > 0.9


def _is_gibberish(text: str) -> bool:
    if not _is_latin(text):
        return False  # never judge Tamil, Hindi... scripts with English heuristics
    words = [w for w in _words(text) if len(w) >= 4]
    if not words:
        return len(re.findall(r"[a-z]", text.lower())) == 0 and len(text.strip()) > 0
    bad = [w for w in words if not re.search(r"[aeiouy]", w) or re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", w)]
    return len(bad) / len(words) >= 0.5


def _is_chit_chat(text: str) -> bool:
    words = _words(text)
    if not words or len(words) > 8 or SUPPORT_CUES.search(text):
        return False
    return all(w in GREETINGS for w in words)


def _recent(log_df: Optional[pd.DataFrame], contact: str, minutes: int) -> pd.DataFrame:
    if log_df is None or log_df.empty or not contact:
        return pd.DataFrame()
    df = log_df[log_df["customer_contact"].str.lower() == contact.lower()].copy()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df[df["ts"] >= datetime.now() - timedelta(minutes=minutes)]


# ----------------------------------------------------------------------------- input
def check_input(text: str, *, contact: str = "", name: str = "", company: str = "Remya Industries Ltd",
                log_df: Optional[pd.DataFrame] = None, followup: bool = False) -> GuardResult:
    """``followup``: the message continues an earlier ticket ("any update?"), so it is never chit-chat."""
    raw = (text or "").strip()
    flags: List[str] = []
    first = f" {name.split()[0]}" if name.strip() else ""

    def block(reason: str, previous: str = "") -> GuardResult:
        template = REPLIES.get(reason)
        reply = template.format(name=first, company=company, previous=previous) if template else None
        return GuardResult(False, reason, LABELS[reason], flags, clean_text, reply, previous)

    clean_text = raw
    if len(clean_text) > MAX_CHARS:
        clean_text = clean_text[:MAX_CHARS] + " ..."
        flags.append("truncated")
    clean_text, pii = mask_pii(clean_text)
    flags += [f"pii_masked:{k}" for k in pii]

    if not re.search(r"\w", clean_text):
        return block("empty")
    has_cue = bool(SUPPORT_CUES.search(clean_text)) or followup

    # duplicates / flooding are only checked for real channel contacts (email, Telegram, WhatsApp)
    recent = _recent(log_df, contact, 15)
    if not recent.empty:
        same = recent[recent["complaint"].str.strip().str.lower() == clean_text.lower()]
        if not same.empty:
            return block("duplicate", same.iloc[-1]["ticket_id"])
        if len(_recent(log_df, contact, 10)) >= 5:
            return block("flooding", recent.iloc[-1]["ticket_id"])

    if SPAM_PATTERNS.search(clean_text) or (URL.search(clean_text) and not has_cue):
        return block("spam")
    if INJECTION.search(clean_text):
        if not has_cue:
            return block("prompt_injection")
        flags.append("prompt_injection")  # real issue + manipulation attempt -> handled, always escalated
    if _is_chit_chat(clean_text) and not followup:
        return block("chit_chat")
    if _is_gibberish(clean_text):
        return block("gibberish")
    if ABUSE.search(clean_text):
        if not has_cue:
            return block("abuse")
        flags.append("abusive_language")
    return GuardResult(True, "", "", flags, clean_text, None)


# ----------------------------------------------------------------------------- output
_META_LINES = re.compile(
    r"^\s*(as an ai|i am an ai|i'm an ai|as a language model|note to (self|agent)|internal note|"
    r"\(internal|knowledge base agent|resolution agent|customer query agent|log agent|support manager|"
    r"system prompt|key_facts|context_brief|terminate|logged\b|route:)", re.I)


def sanitize_reply(text: str, max_words: int = 230) -> tuple[str, List[str]]:
    """Remove AI chatter, internal notes and debris from a customer reply."""
    removed: List[str] = []
    out = text or ""
    if "```" in out:
        out = re.sub(r"```[a-z]*", "", out)
        removed.append("code fences")
    lines = []
    for line in out.splitlines():
        if _META_LINES.search(line):
            removed.append(f"meta line: {line.strip()[:60]}")
            continue
        lines.append(line)
    out = "\n".join(lines)
    new = re.sub(r"(?m)^\s*#{1,6}\s*", "", out)
    new = new.replace("**", "").replace("__", "")
    if new != out:
        removed.append("markdown formatting")
    out = re.sub(r"\n{3,}", "\n\n", new).strip()
    words = out.split()
    if len(words) > max_words:
        cut = " ".join(words[:max_words])
        cut = cut[: cut.rfind(".") + 1] if "." in cut else cut
        out, _ = cut, removed.append(f"trimmed to {max_words} words")
    return out, removed


def describe(result: GuardResult) -> Dict:
    return {"allowed": result.allowed, "reason": result.reason, "label": result.label, "flags": result.flags,
            "previous_ticket": result.previous_ticket}


STRONG_BLOCKS = {"spam", "prompt_injection", "abuse"}


def manager_says_none(manager_text: str, complaint: str) -> bool:
    """True when the Support Manager's ROUTE: none should stop the desk.

    Strong reasons (spam, manipulation, abuse) always stop it. Softer ones (chit-chat, off-topic) are
    overruled when the customer's text clearly mentions an order, product, payment or service - a real
    complaint must never be thrown away; it goes to the general group and on to a human instead.
    """
    if not re.search(r"ROUTE:\s*`?none\b", manager_text or "", flags=re.I):
        return False
    m = re.search(r'"guardrail_reason"\s*:\s*"([a-z_]+)"', manager_text or "")
    reason = m.group(1) if m else "off_topic"
    return reason in STRONG_BLOCKS or not SUPPORT_CUES.search(complaint or "")
