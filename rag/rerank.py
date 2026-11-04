"""Second-stage rerankers, applied to the fused BM25 + vector candidates.

Hybrid search (reciprocal rank fusion) is good at *recall*: the right passage is almost always among the
top ~12 candidates. A reranker reads each candidate against the question and fixes the *order*, so the
passage that answers the question is the one the agents see first.

``RERANKER`` in .env:

* ``feature``        (default, offline) - scores each candidate on signals the fusion score ignores: how
  much of the question's key vocabulary the passage covers (IDF-weighted), whether the section title
  names the topic, exact multi-word phrase matches, and the semantic similarity.
* ``cross-encoder``  - a neural cross-encoder that reads question + passage together. The default model
  ``BAAI/bge-reranker-v2-m3`` is multilingual (Tamil, Hindi, ...). Needs ``pip install sentence-transformers``
  (downloads the model once); falls back to ``feature`` if it cannot load.
* ``none``           - keep the fusion order (used by the evaluation's ablation study).
"""
from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from rag.text import content_terms, expand_query, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from rag.retriever import Hit, HybridIndex

log = logging.getLogger(__name__)
_CROSS_ENCODERS: Dict[str, object] = {}
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


def _bigrams(tokens: List[str]) -> set:
    return {(a, b) for a, b in zip(tokens, tokens[1:])}


# Feature weights of the offline reranker (tuned on the dev split of evaluation/retrieval_set.json,
# checked on the held-out test split). The fusion prior keeps what hybrid search already got right; the
# other features move a passage up when it covers the question's key words, names the topic in its
# section title or shares exact phrases with the question.
FEATURE_WEIGHTS = {"prior": 0.30, "coverage": 0.15, "focus": 0.15, "title": 0.15, "phrases": 0.10, "semantic": 0.15}


def question_focus(text: str) -> str:
    """The sentence(s) that ask the question ("... Is it covered under warranty?") - they state the need, the
    other sentences describe the situation. Empty when the message asks nothing explicitly."""
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip().endswith("?"))


def feature_scores(index: "HybridIndex", queries: List[str], hits: List["Hit"],
                   weights: Optional[Dict[str, float]] = None) -> List[float]:
    """Offline reranker: fusion prior + key-term coverage + section-title match + phrase overlap + similarity."""
    w = weights or FEATURE_WEIGHTS
    q_tokens = [tokenize(expand_query(q)) for q in queries]
    focus_terms = set(content_terms(tokenize(expand_query(question_focus(queries[0]))))) if queries else set()
    top_rel = max((h.relevance for h in hits), default=0.0) or 1.0
    scores = []
    for h in hits:
        title_tokens = set(tokenize(h.chunk.section))
        chunk_tokens = index.tokens_of(h.chunk)
        doc_terms = set(chunk_tokens) | title_tokens
        focus = (sum(index.idf(t) for t in focus_terms if t in doc_terms) / sum(index.idf(t) for t in focus_terms)
                 if focus_terms else h.coverage)
        title = phrases = 0.0
        for toks in q_tokens:
            key = set(content_terms(toks))
            if key:
                title = max(title, sum(index.idf(t) for t in key if t in title_tokens) / sum(index.idf(t) for t in key))
            phrases = max(phrases, len(_bigrams(toks) & _bigrams(chunk_tokens)) / max(1, len(_bigrams(toks))))
        prior = 1 / math.sqrt(max(1, h.fusion_rank))
        scores.append(w["prior"] * prior + w["coverage"] * h.coverage + w["focus"] * focus + w["title"] * title
                      + w["phrases"] * phrases + w["semantic"] * (h.relevance / top_rel))
    return scores


def _cross_encoder(model_name: str):
    if model_name not in _CROSS_ENCODERS:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            _CROSS_ENCODERS[model_name] = CrossEncoder(model_name, max_length=512)
        except Exception as exc:  # remember the failure: warn once, not on every search
            _CROSS_ENCODERS[model_name] = exc
    model = _CROSS_ENCODERS[model_name]
    if isinstance(model, Exception):
        raise model
    return model


def cross_encoder_scores(model_name: str, queries: List[str], hits: List["Hit"]) -> List[float]:
    model = _cross_encoder(model_name or DEFAULT_CROSS_ENCODER)
    best = [float("-inf")] * len(hits)
    for q in queries:  # original and translated query: keep the better score
        raw = model.predict([(q, f"{h.chunk.section}\n{h.chunk.text}") for h in hits])
        for i, r in enumerate(raw):
            best[i] = max(best[i], 1 / (1 + math.exp(-float(r))))  # logits -> 0..1
    return best


def rerank(index: "HybridIndex", queries: List[str], hits: List["Hit"], method: str = "feature",
           model: str = "") -> List["Hit"]:
    """Re-order hits in place of the fusion order; every hit keeps its fusion rank for the trace."""
    method = (method or "feature").lower()
    if not hits or method == "none":
        for h in hits:
            h.rerank_score = None
        return hits
    scores: Optional[List[float]] = None
    if method in {"cross-encoder", "cross_encoder", "ce"}:
        try:
            scores = cross_encoder_scores(model, queries, hits)
            used = f"cross-encoder ({model or DEFAULT_CROSS_ENCODER})"
        except Exception as exc:  # package or model missing -> offline reranker, never crash the desk
            if not getattr(exc, "_warned", False):
                log.warning("Cross-encoder reranker unavailable (%s); using the feature reranker", exc)
                try:
                    exc._warned = True  # type: ignore[attr-defined]
                except Exception:
                    pass
    if scores is None:
        scores = feature_scores(index, queries, hits)
        used = "feature"
    for h, s in zip(hits, scores):
        h.rerank_score = round(float(s), 4)
        h.reranker = used
    return sorted(hits, key=lambda h: (h.rerank_score, h.score), reverse=True)
