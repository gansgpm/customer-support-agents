"""Knowledge store with three RAG collections, scoped by product line / service.

1. **Policy RAG**   – ``knowledge_base/general|product_lines|services/*.md`` (Knowledge Base Agent).
   Every chunk carries the product line it belongs to (from ``catalog.json``), so a search for an
   electronics ticket only sees electronics + general policies, never the grocery or fashion rules.
2. **Learned RAG**  – solutions resolved with high confidence or approved by a human, fed back by the
   Log Agent and tagged with their product line (Knowledge Base Agent, for re-use)
3. **History RAG**  – past tickets from ``logs/tickets.csv`` (Customer Query Agent: repeat complaints,
   earlier tickets for the same order / customer and the product line they were about)
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from filelock import FileLock

from engine.catalog import GENERAL, get_catalog
from engine.config import Settings
from rag import multilingual
from rag.index_store import IndexStore, fingerprint
from rag.retriever import Hit, HybridIndex
from rag.text import Chunk

CATEGORY_SOURCES = {
    "Shipping": ["shipping_policy.md"],
    "Refund": ["refund_policy.md"],
    "Account": ["account_and_login.md"],
    "Product": ["product_faq.md"],
    "Billing": ["billing_and_payments.md"],
}
# Specificity boosts: the ticket's own product-line guide outranks a general policy unless the general passage
# is clearly better, and the policy file for the ticket's category outranks other general files.
# (rerank scores are 0..1; plain RRF fusion scores are ~0.03, hence two scales)
BOOSTS = {"reranked": {"line": 0.35, "category": 0.03}, "fusion": {"line": 0.012, "category": 0.01}}

log = logging.getLogger(__name__)
_lock = threading.Lock()
_POLICY_CACHE: Dict[tuple, HybridIndex] = {}
_LEARNED_CACHE: Dict[tuple, HybridIndex] = {}


# Default 'covered' thresholds per embedding backend (cosine similarity means different things for TF-IDF
# and neural vectors). `python -m evaluation.retrieval_eval --calibrate` measures better values on the dev
# questions and writes evaluation/calibration/<backend>.json, which is used automatically.
DEFAULT_THRESHOLDS = {
    "tfidf": {"min_relevance": 0.08, "min_coverage": 0.40, "strong_relevance": 0.35},
    "openai": {"min_relevance": 0.30, "min_coverage": 0.40, "strong_relevance": 0.50},
    "sentence-transformers": {"min_relevance": 0.35, "min_coverage": 0.40, "strong_relevance": 0.55},
}
STRONG_MIN_COVERAGE = 0.25  # a very close semantic match may leave this much customer wording unmatched
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "calibration"


def calibration_file(backend: str, model: str = "") -> Path:
    name = backend + (f"__{model.replace('/', '_')}" if model else "")
    return CALIBRATION_DIR / f"{name}.json"


def is_covered(top_relevance: float, coverage: float, th: Dict[str, float]) -> bool:
    """The knowledge base covers the question when the best passage is relevant enough AND the passages contain
    enough of the question's key words - or the semantic match is very close."""
    return bool(top_relevance >= th["min_relevance"] and (
        coverage >= th["min_coverage"] or (top_relevance >= th["strong_relevance"] and coverage >= STRONG_MIN_COVERAGE)))


def _kb_files(kb_dir: Path) -> List[Path]:
    return sorted(p for p in kb_dir.rglob("*.md") if "learned" not in p.relative_to(kb_dir).parts)


def _dir_signature(kb_dir: Path) -> tuple:
    files = _kb_files(kb_dir) + [kb_dir / "catalog.json"]
    return tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.exists())


def _file_signature(path: Path) -> tuple:
    return (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (0, 0)


def parse_lines(product_line) -> List[str]:
    """'electronics, installation_repair' / ['electronics'] -> ['electronics', 'installation_repair']."""
    if not product_line:
        return []
    items: Iterable = product_line if isinstance(product_line, (list, tuple)) else str(product_line).split(",")
    return [str(i).strip().lower() for i in items if str(i).strip()]


class KnowledgeStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.kb_dir = Path(settings.kb_dir)
        self.learned_file = Path(settings.learned_file)

    @property
    def catalog(self):
        return get_catalog(self.kb_dir)

    # ------------------------------------------------------------------ policy
    def thresholds(self) -> Dict:
        """'Covered' thresholds: explicit .env values > calibration file for this backend > backend defaults."""
        s = self.settings
        backend = s.embedding_backend if s.embedding_backend in DEFAULT_THRESHOLDS else "tfidf"
        th = dict(DEFAULT_THRESHOLDS[backend])
        source = f"default for {backend}"
        cal = calibration_file(s.embedding_backend, s.embedding_model)
        if not cal.exists():
            cal = calibration_file(s.embedding_backend)
        if cal.exists():
            try:
                data = json.loads(cal.read_text(encoding="utf-8"))
                th.update({k: float(data[k]) for k in th if k in data})
                source = f"calibrated ({cal.name}, {data.get('calibrated_at', '')})"
            except (OSError, ValueError, KeyError):
                pass
        explicit = {"min_relevance": s.kb_min_relevance, "min_coverage": s.kb_min_coverage,
                    "strong_relevance": s.kb_strong_relevance}
        if any(v is not None for v in explicit.values()):
            th.update({k: v for k, v in explicit.items() if v is not None})
            source = "set in .env"
        return {**th, "source": source}

    @property
    def index_store(self) -> IndexStore:
        return IndexStore(Path(self.settings.data_dir) / "index" / "policy")

    def policy_index(self) -> HybridIndex:
        """Hybrid index over the policy documents: in memory per process, saved on disk between runs."""
        from rag.text import chunk_markdown

        s = self.settings
        key = (str(self.kb_dir), s.embedding_backend, s.embedding_model, _dir_signature(self.kb_dir))
        with _lock:
            if key not in _POLICY_CACHE:
                files = _kb_files(self.kb_dir)
                expected = fingerprint(files, self.kb_dir, s.embedding_backend, s.embedding_model)
                store = self.index_store
                saved = store.load(expected)
                if saved:  # unchanged documents and settings: reuse the saved chunks and embeddings
                    chunks, vectors = saved
                    index = HybridIndex(chunks, s.embedding_backend, s.embedding_model, vectors=vectors)
                    index.origin = "loaded from disk"
                else:
                    catalog = self.catalog
                    chunks = []
                    for path in files:
                        scope = catalog.scope_of(path.name)
                        for c in chunk_markdown(path, scope=scope):
                            c.meta["division"] = catalog.name(scope)
                            chunks.append(c)
                    index = HybridIndex(chunks, s.embedding_backend, s.embedding_model)
                    index.origin = "built"
                    try:
                        store.save(chunks, index.vectors, {**expected,
                                                           "embedder": getattr(index, "embedder_name", "")})
                    except OSError as exc:  # read-only folder: keep working from memory
                        log.warning("Could not save the knowledge index: %s", exc)
                _POLICY_CACHE.clear()
                _POLICY_CACHE[key] = index
            return _POLICY_CACHE[key]

    def rebuild_index(self) -> HybridIndex:
        with _lock:
            _POLICY_CACHE.clear()
        self.index_store.clear()
        return self.policy_index()

    def index_info(self) -> Dict:
        index = self.policy_index()
        manifest = self.index_store.manifest()
        return {"origin": getattr(index, "origin", "built"), "chunks": len(index.chunks),
                "documents": len(manifest.get("files", {})), "built_at": manifest.get("built_at", ""),
                "embedding": f"{self.settings.embedding_backend} {self.settings.embedding_model}".strip(),
                "chunker": manifest.get("chunker", ""), "vector_store": index.vector_store,
                "dimensions": manifest.get("dimensions", 0), "folder": str(self.index_store.folder)}

    def documents(self) -> List[Path]:
        return _kb_files(self.kb_dir)

    def documents_by_scope(self) -> Dict[str, List[Path]]:
        out: Dict[str, List[Path]] = {}
        for p in self.documents():
            out.setdefault(self.catalog.scope_of(p.name), []).append(p)
        return out

    # ----------------------------------------------------------------- learned
    def learned_entries(self) -> List[Dict]:
        if not self.learned_file.exists():
            return []
        rows = []
        for line in self.learned_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def learned_index(self) -> HybridIndex:
        key = (str(self.learned_file), _file_signature(self.learned_file))
        with _lock:
            if key not in _LEARNED_CACHE:
                chunks = [
                    Chunk(
                        id=f"learned#{e['ticket_id']}",
                        source=f"learned:{e['ticket_id']}",
                        section=e.get("question", ""),
                        text=e.get("answer", ""),
                        collection="learned",
                        meta={"search_text": e.get("question", ""),  # match on the customer's question only
                              "ticket_id": e["ticket_id"], "category": e.get("category", ""),
                              "scope": e.get("product_line", GENERAL),
                              "reuse_count": e.get("reuse_count", 0), "policy_sources": e.get("sources", "")},
                    )
                    for e in self.learned_entries()
                ]
                _LEARNED_CACHE.clear()
                # learned answers always use offline vectors: cheap to rebuild after every new solution
                _LEARNED_CACHE[key] = HybridIndex(chunks, "tfidf")
            return _LEARNED_CACHE[key]

    def add_learned_solution(self, *, ticket_id: str, category: str, question: str, answer: str,
                             sources: str, confidence: int, approved_by: str = "auto",
                             product_line: str = GENERAL) -> Dict:
        """Feed a verified answer back into the Knowledge Base Agent's learned RAG.

        Near-duplicate questions (same category and product line) update the existing entry
        (reuse_count += 1) instead of adding noise to the index.
        """
        product_line = product_line or GENERAL
        self.learned_file.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.learned_file) + ".lock"):
            entries = self.learned_entries()
            dup = None
            if entries:
                hits = self.learned_index().search(question, k=1)
                if hits and hits[0].relevance >= 0.6 and hits[0].chunk.meta.get("category") == category \
                        and hits[0].chunk.meta.get("scope", GENERAL) == product_line:
                    dup = hits[0].chunk.meta["ticket_id"]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if dup:
                for e in entries:
                    if e["ticket_id"] == dup:
                        e["reuse_count"] = int(e.get("reuse_count", 0)) + 1
                        e["last_seen"] = now
                        if approved_by == "human":
                            e["answer"], e["approved_by"] = answer, "human"
                action = {"action": "updated", "ticket_id": dup}
            else:
                entries.append({"ticket_id": ticket_id, "created_at": now, "last_seen": now, "category": category,
                                "product_line": product_line, "question": question, "answer": answer,
                                "sources": sources, "confidence": confidence, "approved_by": approved_by,
                                "reuse_count": 0})
                action = {"action": "added", "ticket_id": ticket_id}
            tmp = self.learned_file.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
            tmp.replace(self.learned_file)
        return action

    # --------------------------------------------------------------- search_kb
    def search_kb(self, query: str, category: str = "", product_line="", k: Optional[int] = None,
                  english_hint: str = "", reranker: Optional[str] = None, mode: str = "hybrid",
                  translate: bool = True, also: Sequence[str] = ()) -> Dict:
        """Hybrid search, scoped to the ticket's product line(s) + general policies, then reranked.

        With no product line (or 'general') every document is searchable. When the query is not English
        (Tamil, Hindi, ...) it is searched together with an English version: ``english_hint`` (the Query
        Agent's English summary) and the offline glossary translation - all fused into one ranking.
        """
        s = self.settings
        k = k or s.kb_top_k
        reranker = s.reranker if reranker is None else reranker
        lines = [l for l in parse_lines(product_line) if l != GENERAL and l in self.catalog.ids()]
        allowed = set(lines) | {GENERAL}
        in_scope = (lambda c: c.meta.get("scope", GENERAL) in allowed) if lines else None
        queries = multilingual.search_queries(query, english_hint) if translate else [query]
        queries += [q for q in also if q and q.strip() and q not in queries]  # e.g. the customer's own words
        extra = queries[1:]

        index = self.policy_index()
        hits = index.search(query, k=max(k + 3, s.kb_candidates), where=in_scope, extra_queries=extra,
                            candidates=s.kb_candidates, rrf_k=s.rrf_k, reranker=reranker,
                            rerank_model=s.reranker_model, mode=mode)
        preferred = CATEGORY_SOURCES.get((category or "").strip().title(), [])
        reranked = any(h.rerank_score is not None for h in hits)

        boost = BOOSTS["reranked" if reranked else "fusion"]

        home = set(lines) if lines else {GENERAL}  # the ticket's own division(s); general tickets -> general policies
        scores: Dict[int, float] = {}

        def rank(h: Hit) -> float:
            base = h.rerank_score if h.rerank_score is not None else h.score
            bonus = boost["line"] if h.chunk.meta.get("scope", GENERAL) in home else 0.0
            if h.chunk.source in preferred:
                bonus += boost["category"]
            scores[id(h)] = base + bonus
            return base + bonus

        hits.sort(key=rank, reverse=True)
        if len(lines) > 1 and hits:  # multi-division ticket: each division's best passage up front, when relevant
            best = scores[id(hits[0])]
            firsts = [next((h for h in hits if h.chunk.meta.get("scope") == line), None) for line in lines]
            firsts = [h for h in firsts if h is not None and scores[id(h)] >= 0.5 * best]
            firsts.sort(key=lambda h: scores[id(h)], reverse=True)
            hits = firsts + [h for h in hits if h not in firsts]
        hits = hits[:k]
        learned_hits = [h for h in self.learned_index().search(query, k=2, where=in_scope, extra_queries=extra)
                        if h.relevance >= s.learned_min_relevance]

        top = max(hits[:2], key=lambda h: h.relevance) if hits else None
        coverage = index.union_coverage(query, hits[:max(3, len(lines))], extra_queries=extra) if hits else 0.0
        th = self.thresholds()
        policy_match = bool(top) and is_covered(top.relevance, coverage, th)
        learned_match = bool(learned_hits and learned_hits[0].coverage >= 0.5)
        return {
            "match_found": policy_match or learned_match,
            "policy_match": policy_match,
            "learned_match": learned_match,
            "scoped_to": sorted(allowed) if lines else ["all documents"],
            "top_relevance": round(top.relevance, 3) if top else 0.0,
            "top_coverage": round(coverage, 3),
            "thresholds": th,
            "query_language": multilingual.detect_language(query),
            "search_queries": queries,
            "policy_passages": [{**h.to_dict(), "rank_score": round(scores.get(id(h), h.score), 4)}
                                for h in hits if h.relevance > 0.02 or h.coverage > 0],
            "resolved_ticket_matches": [h.to_dict(600) for h in learned_hits],
            "retriever": (f"hybrid BM25 + {index.backend_name} vectors ({index.vector_store}), RRF over "
                          f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'}, "
                          f"reranker: {hits[0].reranker if hits and hits[0].reranker else 'none'}"),
        }

    # ---------------------------------------------------------- ticket history
    def search_ticket_history(self, query: str, order_id: str = "", customer_contact: str = "",
                              exclude_ticket: str = "", k: int = 3) -> Dict:
        empty = {"similar_tickets": [], "same_customer_or_order": [], "repeat_issue": False, "history_size": 0,
                 "product_lines_seen_for_order": []}
        log_file = Path(self.settings.log_file)
        if not log_file.exists():
            return empty
        try:
            df = pd.read_csv(log_file, dtype=str, keep_default_na=False).fillna("")
        except Exception:
            return empty
        if exclude_ticket:
            df = df[df["ticket_id"] != exclude_ticket]
        if "status" in df.columns:  # messages stopped by the guardrail are not support history
            df = df[df["status"] != "Filtered"]
        if df.empty:
            return empty
        if "product_line" not in df.columns:
            df["product_line"] = ""

        chunks = [
            Chunk(id=r["ticket_id"], source=f"ticket:{r['ticket_id']}", section=r.get("summary", ""),
                  text=r.get("complaint", ""), collection="tickets",
                  meta={"ticket_id": r["ticket_id"], "category": r.get("category", ""), "status": r.get("status", ""),
                        "product_line": r.get("product_line", "")})
            for _, r in df.iterrows()
        ]
        hits: List[Hit] = [h for h in HybridIndex(chunks, "tfidf").search(query, k=k) if h.relevance >= 0.25]
        similar = [{"ticket_id": h.chunk.meta["ticket_id"], "category": h.chunk.meta["category"],
                    "product_line": h.chunk.meta["product_line"], "status": h.chunk.meta["status"],
                    "relevance": round(h.relevance, 2), "summary": h.chunk.section or h.chunk.text[:120]} for h in hits]

        same = []
        if order_id or customer_contact:
            mask = pd.Series(False, index=df.index)
            if order_id:
                mask |= df["order_id"].str.strip().str.lower() == order_id.strip().lower()
            if customer_contact:
                mask |= df["customer_contact"].str.strip().str.lower() == customer_contact.strip().lower()
            for _, r in df[mask].tail(5).iterrows():
                same.append({"ticket_id": r["ticket_id"], "category": r["category"], "status": r["status"],
                             "product_line": r.get("product_line", ""), "summary": r.get("summary", "")[:120]})
        order_lines = []
        if order_id:
            order_lines = sorted({r["product_line"] for r in same if r["product_line"] and r["product_line"] != GENERAL})
        return {
            "similar_tickets": similar,
            "same_customer_or_order": same,
            "repeat_issue": bool(same),
            "history_size": int(len(df)),
            "product_lines_seen_for_order": order_lines,
        }
