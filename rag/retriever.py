"""Hybrid retriever: BM25 keyword search + dense vector search (FAISS), fused with RRF, then reranked.

Pipeline for one search: every query (the customer's words, plus an English version when they wrote in
Tamil, Hindi, ...) is run through BM25 and through the vector index; all rankings are merged with
reciprocal rank fusion into a candidate pool (``KB_CANDIDATES``), a reranker re-orders the pool
(``RERANKER``, see rag/rerank.py) and the top ``KB_TOP_K`` passages go to the agent. Every hit records
which retriever found it (``found_by``), its BM25 / vector / fusion ranks and its rerank score, so the UI,
the trace log and the evaluation can show *why* a passage was chosen.

Vector backends
---------------
* ``tfidf``                 – default, fully offline (scikit-learn TF-IDF vectors stored in FAISS)
* ``openai``                – OpenAI ``text-embedding-3-small`` (needs OPENAI_API_KEY)
* ``sentence-transformers`` – local ``paraphrase-multilingual-MiniLM-L12-v2`` (multilingual: Tamil, Hindi, ...;
  needs the package + model download)

Besides a ranking, every hit carries two *absolute* signals used to decide
whether the knowledge base actually covers a question:

* ``relevance`` – cosine similarity of the query and the chunk (0..1)
* ``coverage``  – IDF-weighted share of the query's key terms found in the chunk.
  Terms that never appear in the knowledge base get the highest weight, so a
  question about something the company never wrote about ("gift cards") scores low.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from rag.text import Chunk, content_terms, expand_query, tokenize

log = logging.getLogger(__name__)

try:  # FAISS is optional – numpy is used as a fallback
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    HAS_FAISS = False


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score (ranking only)
    relevance: float  # cosine similarity 0..1
    coverage: float  # weighted query-term coverage 0..1
    bm25: float = 0.0
    bm25_rank: Optional[int] = None  # 1-based rank in the BM25 list (best over all queries), None = not found
    vector_rank: Optional[int] = None  # 1-based rank in the vector list
    fusion_rank: int = 0
    rerank_score: Optional[float] = None
    reranker: str = ""
    found_by: List[str] = field(default_factory=list)  # ["bm25", "vector"]

    def to_dict(self, max_chars: int = 900) -> Dict:
        text = self.chunk.text if len(self.chunk.text) <= max_chars else self.chunk.text[:max_chars] + " ..."
        out = {
            "source": self.chunk.source,
            "section": self.chunk.section,
            "relevance": round(self.relevance, 3),
            "coverage": round(self.coverage, 3),
            "text": text,
            "chunk_id": self.chunk.meta.get("chunk_id", self.chunk.id),
            "found_by": "+".join(self.found_by) or "-",
            "ranks": {"bm25": self.bm25_rank, "vector": self.vector_rank, "fusion": self.fusion_rank},
            "rerank_score": self.rerank_score,
        }
        out.update({k: v for k, v in self.chunk.meta.items()
                    if k in {"ticket_id", "category", "status", "reuse_count", "scope", "doc_title", "doc_type"}})
        return out


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------
class _TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                                    sublinear_tf=True, min_df=1)

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        return self._vec.fit_transform(texts).toarray().astype("float32")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype("float32")


_QUERY_CACHE: Dict[tuple, np.ndarray] = {}


class _OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model or "text-embedding-3-small"

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):  # batched: one request per 256 chunks
            resp = self._client.embeddings.create(model=self._model, input=list(texts[i:i + 256]))
            out.extend(d.embedding for d in resp.data)
        return np.array(out, dtype="float32")

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Query embeddings are cached: the same question (or its rewrite) is never paid for twice."""
        missing = [t for t in texts if (self._model, t) not in _QUERY_CACHE]
        if missing:
            for t, vec in zip(missing, self._embed(missing)):
                if len(_QUERY_CACHE) > 5000:
                    _QUERY_CACHE.clear()
                _QUERY_CACHE[(self._model, t)] = vec
        return np.array([_QUERY_CACHE[(self._model, t)] for t in texts], dtype="float32")


class _SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model: str = "") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model or MULTILINGUAL_ST_MODEL)

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True), dtype="float32")


MULTILINGUAL_ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 50+ languages incl. Tamil and Hindi


def make_embedder(backend: str, model: str = ""):
    backend = (backend or "tfidf").lower()
    try:
        if backend == "openai":
            return _OpenAIEmbedder(model)
        if backend in {"sentence-transformers", "st", "local"}:
            return _SentenceTransformerEmbedder(model)  # default: multilingual MiniLM
    except Exception as exc:  # fall back to offline vectors rather than crash the desk
        log.warning("Embedding backend %s unavailable (%s); falling back to TF-IDF", backend, exc)
    return _TfidfEmbedder()


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ---------------------------------------------------------------------------
# Hybrid index
# ---------------------------------------------------------------------------
class HybridIndex:
    """BM25 + vector index over one collection of chunks."""

    def __init__(self, chunks: List[Chunk], backend: str = "tfidf", embedding_model: str = "",
                 embedder_factory: Optional[Callable] = None, vectors: Optional[np.ndarray] = None) -> None:
        """``vectors``: embeddings saved by rag/index_store.py - reused instead of embedding the chunks again."""
        self.chunks = chunks
        self.backend_name = backend
        self.embedding_model = embedding_model
        self._pos = {id(c): i for i, c in enumerate(chunks)}
        texts = [expand_query(c.searchable) for c in chunks]  # same normalisation as queries
        self._tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokens) if chunks else None
        # document frequency for coverage weighting
        n = max(len(chunks), 1)
        df: Dict[str, int] = {}
        for toks in self._tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self._idf_unknown = max(self._idf.values()) if self._idf else 1.0
        self._embedder = None
        self._index = None
        self._matrix = None
        if chunks:
            self._embedder = (embedder_factory or make_embedder)(backend, embedding_model)
            reuse = vectors is not None and getattr(self._embedder, "name", "") != "tfidf" \
                and vectors.shape[0] == len(chunks)
            # TF-IDF must (re)learn its vocabulary - cheap; neural embeddings are reused from disk when saved
            mat = vectors.astype("float32") if reuse else _normalize(self._embedder.fit(texts))
            self._matrix = mat
            if HAS_FAISS:
                self._index = faiss.IndexFlatIP(mat.shape[1])
                self._index.add(mat)

    @property
    def vector_store(self) -> str:
        return "FAISS" if self._index is not None else "numpy"

    @property
    def vectors(self) -> Optional[np.ndarray]:
        return self._matrix

    def tokens_of(self, chunk: Chunk) -> List[str]:
        return self._tokens[self._pos[id(chunk)]]

    def idf(self, token: str) -> float:
        return self._idf.get(token, self._idf_unknown)

    # ---------------------------------------------------------------------
    def _coverage(self, query_tokens: List[str], chunk_tokens) -> float:
        query_tokens = content_terms(query_tokens)
        if not query_tokens:
            return 0.0
        doc = set(chunk_tokens)
        total = matched = 0.0
        for t in set(query_tokens):
            w = self._idf.get(t, self._idf_unknown)
            total += w
            if t in doc:
                matched += w
        return matched / total if total else 0.0

    def _cosine(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        q_vec = _normalize(self._embedder.encode([query]))
        if self._index is not None:
            sims, ids = self._index.search(q_vec, n)
            cos = np.zeros(n, dtype="float32")
            cos[ids[0]] = sims[0]
            return cos
        return (self._matrix @ q_vec.T).ravel()

    def search(self, query: str, k: int = 4, where: Optional[Callable[[Chunk], bool]] = None,
               extra_queries: Sequence[str] = (), candidates: int = 0, rrf_k: int = 60,
               reranker: str = "none", rerank_model: str = "", mode: str = "hybrid") -> List[Hit]:
        """Hybrid search.

        ``extra_queries``: more phrasings of the same need (e.g. the English translation) - fused with RRF.
        ``candidates``: size of the fused pool handed to the reranker (default: k).
        ``mode``: ``hybrid`` | ``bm25`` | ``vector`` (single retrievers, used by the evaluation's ablation).
        """
        queries = [q for q in dict.fromkeys([query, *extra_queries]) if q and q.strip()]
        if not self.chunks or not queries:
            return []
        allowed = [i for i, c in enumerate(self.chunks) if where is None or where(c)]
        if not allowed:
            return []
        pool = max(k, candidates or k)
        n = len(self.chunks)
        best_bm25 = np.zeros(n)
        best_cos = np.full(n, -1.0)
        best_cov = np.zeros(n)
        bm25_rank: Dict[int, int] = {}
        vec_rank: Dict[int, int] = {}
        fused = {i: 0.0 for i in allowed}
        for q in queries:
            q = expand_query(q)
            q_tokens = tokenize(q)
            bm25_scores = np.array(self._bm25.get_scores(q_tokens)) if q_tokens else np.zeros(n)
            cos = self._cosine(q)
            best_bm25 = np.maximum(best_bm25, bm25_scores)
            best_cos = np.maximum(best_cos, cos)
            for i in allowed:
                best_cov[i] = max(best_cov[i], self._coverage(q_tokens, self._tokens[i]))
            rankings = []
            if mode in {"hybrid", "bm25"}:
                rankings.append(("bm25", [i for i in sorted(allowed, key=lambda i: -bm25_scores[i])
                                          if bm25_scores[i] > 0]))
            if mode in {"hybrid", "vector"}:
                rankings.append(("vector", [i for i in sorted(allowed, key=lambda i: -cos[i]) if cos[i] > 0]))
            for name, ranking in rankings:  # reciprocal rank fusion over every query x retriever ranking
                ranks = bm25_rank if name == "bm25" else vec_rank
                for r, i in enumerate(ranking):
                    fused[i] += 1 / (rrf_k + r + 1)
                    ranks[i] = min(ranks.get(i, r + 1), r + 1)

        order = [i for i in sorted(fused, key=fused.get, reverse=True) if fused[i] > 0]
        hits: List[Hit] = []
        for pos, i in enumerate(order[:pool]):
            found = [name for name, ranks in (("bm25", bm25_rank), ("vector", vec_rank))
                     if ranks.get(i, 10 ** 9) <= pool]
            hits.append(Hit(chunk=self.chunks[i], score=float(fused[i]), relevance=float(max(best_cos[i], 0.0)),
                            coverage=float(best_cov[i]), bm25=float(best_bm25[i]), bm25_rank=bm25_rank.get(i),
                            vector_rank=vec_rank.get(i), fusion_rank=pos + 1, found_by=found))
        if reranker and reranker != "none" and mode == "hybrid":
            from rag.rerank import rerank

            hits = rerank(self, queries, hits, method=reranker, model=rerank_model)
        return hits[:k]

    def union_coverage(self, query: str, hits: List[Hit], extra_queries: Sequence[str] = ()) -> float:
        """Coverage of the query's key terms by the union of the retrieved chunks (best over all phrasings)."""
        tokens = set()
        for h in hits:
            tokens.update(self.tokens_of(h.chunk))
        return max((self._coverage(tokenize(expand_query(q)), tokens)
                    for q in dict.fromkeys([query, *extra_queries]) if q and q.strip()), default=0.0)
