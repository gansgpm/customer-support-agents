"""Ground-truth retrieval evaluation: hit-rate@k, MRR and abstention accuracy, with an ablation study.

    python -m evaluation.retrieval_eval                       # EMBEDDING_BACKEND / RERANKER / LLM from .env
    python -m evaluation.retrieval_eval --embedding openai    # OpenAI text-embedding-3-small (needs OPENAI_API_KEY)
    python -m evaluation.retrieval_eval --calibrate           # tune the 'covered' thresholds on the DEV split
    python -m evaluation.retrieval_eval --no-llm              # skip the LLM variants (no API calls)

Runs every hand-labelled question in evaluation/retrieval_set.json (English, Tamil, Hindi and questions the
documents do not cover) through the knowledge store, once per retrieval variant:

* BM25 only, vector only, hybrid (RRF fusion), hybrid + reranker
* hybrid + reranker without the multilingual query step (shows what translation adds for Tamil/Hindi)
* with a real LLM (anything but LLM_PROVIDER=offline) the production path of the agents:
  - **LLM query rewrite** - the model turns the customer's message into an English search query, as the
    Knowledge Base Agent does, and it is searched together with the customer's own words;
  - **LLM coverage verdict** - the model reads the retrieved passages and decides covered / not covered,
    as the Knowledge Base Agent's ``match_found`` does. LLM answers are cached in
    evaluation/reports/llm_cache.json, so re-runs cost nothing.

Metrics (answerable questions): hit@1/3/5 = share whose labelled passage is in the top 1/3/5; MRR@10 = mean of
1/rank of the first labelled passage; 95% confidence intervals by bootstrap. Abstention (all questions): covered
must be true for answerable and false for unanswerable questions.

Integrity: the *dev* split is the only data used for tuning (query expansion, reranker weights, thresholds).
The *held-out test* split is fingerprinted (SHA-256 in retrieval_set.json); the report says whether it is
unchanged, and the evaluation checks that no held-out question was copied into the code or the documents.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine.config import BASE_DIR, Settings, get_settings
from rag.knowledge import STRONG_MIN_COVERAGE, KnowledgeStore, calibration_file

SET_FILE = BASE_DIR / "evaluation" / "retrieval_set.json"
REPORT_DIR = BASE_DIR / "evaluation" / "reports"
LLM_CACHE = REPORT_DIR / "llm_cache.json"
TOP_N = 10
SPLIT_NAMES = {"test": "held-out test", "dev": "dev (tuning)", "all": "all"}

REWRITE_PROMPT = """You are the Knowledge Base Agent of a customer-support desk. Rewrite the customer's message as ONE
short English search query for the company's policy documents: translate it if needed, keep the product, the
problem and what the customer wants, use the plain words a policy document would use. Return only the query."""
VERDICT_PROMPT = """You are the Knowledge Base Agent of a customer-support desk. Decide whether the PASSAGES (company
policy documents) contain the answer to the CUSTOMER'S MESSAGE. Loosely related text does not count: the passages
must actually address what the customer asks. Answer with ONLY a JSON object: {"covered": true or false}"""


# ------------------------------------------------------------------ data + integrity
def load_set(path: Path = SET_FILE) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def split_fingerprint(questions: List[Dict], split: str = "test") -> str:
    """SHA-256 of the held-out questions and their labels - changes if anyone edits the held-out split."""
    canon = [{"id": q["id"], "question": q["question"], "expected": q.get("expected", []),
              "answerable": q.get("answerable", True), "scope": q.get("scope", "general")}
             for q in questions if q.get("split", "dev") == split]
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def frozen_fingerprint(path: Path = SET_FILE) -> str:
    return json.loads(path.read_text(encoding="utf-8")).get("test_split_sha256", "")


def leaked_questions(questions: List[Dict]) -> List[str]:
    """Held-out questions whose text appears in the code or the knowledge base (would inflate the results)."""
    corpus = []
    for folder in ("rag", "engine", "agents", "knowledge_base"):
        for p in (BASE_DIR / folder).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".json"}:
                corpus.append(p.read_text(encoding="utf-8", errors="ignore").lower())
    blob = "\n".join(corpus)
    return [q["id"] for q in questions if q.get("split") == "test" and q["question"].lower() in blob]


# ------------------------------------------------------------------ LLM helper (cached)
class LLMHelper:
    """Real-LLM steps of the production path, cached on disk by (model, step, text)."""

    def __init__(self, settings: Settings) -> None:
        from engine.llm import build_model_client, describe

        self.settings = settings
        self.model = describe(settings)
        self.client = build_model_client(settings)
        self.cache: Dict[str, object] = {}
        if LLM_CACHE.exists():
            try:
                self.cache = json.loads(LLM_CACHE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        self.calls = 0

    def _key(self, step: str, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{step}|{text}".encode("utf-8")).hexdigest()

    async def _ask(self, system: str, user: str) -> str:
        from autogen_core.models import SystemMessage, UserMessage

        self.calls += 1
        result = await self.client.create([SystemMessage(content=system), UserMessage(content=user, source="eval")])
        return result.content if isinstance(result.content, str) else str(result.content)

    async def rewrite(self, question: str) -> str:
        key = self._key("rewrite", question)
        if key not in self.cache:
            text = (await self._ask(REWRITE_PROMPT, question)).strip().strip('"').splitlines()[0][:300]
            self.cache[key] = text
        return str(self.cache[key])

    async def verdict(self, question: str, passages: List[Dict]) -> bool:
        evidence = "\n\n".join(f"[{i + 1}] {p['source']} > {p['section']}\n{p['text']}" for i, p in enumerate(passages[:4]))
        key = self._key("verdict", question + "||" + "|".join(p.get("chunk_id", p["section"]) for p in passages[:4]))
        if key not in self.cache:
            from engine.parsing import extract_json

            raw = await self._ask(VERDICT_PROMPT, f"CUSTOMER'S MESSAGE:\n{question}\n\nPASSAGES:\n{evidence or '(none)'}")
            obj = extract_json(raw) or {}
            self.cache[key] = bool(obj.get("covered", "true" in raw.lower()))
        return bool(self.cache[key])

    def save(self) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        LLM_CACHE.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8")

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass


# ------------------------------------------------------------------ variants
def variants(settings: Settings, llm: bool) -> List[Dict]:
    prod = settings.reranker if settings.reranker != "none" else "feature"
    out = [
        {"name": "BM25 only", "mode": "bm25", "reranker": "none", "translate": True},
        {"name": "Vector only", "mode": "vector", "reranker": "none", "translate": True},
        {"name": "Hybrid (RRF)", "mode": "hybrid", "reranker": "none", "translate": True},
        {"name": f"Hybrid + {prod} reranker", "mode": "hybrid", "reranker": prod, "translate": True,
         "production": not llm},
        {"name": "Hybrid + reranker, no translation", "mode": "hybrid", "reranker": prod, "translate": False},
    ]
    if llm:
        out += [
            {"name": "+ LLM query rewrite", "mode": "hybrid", "reranker": prod, "translate": True, "rewrite": True},
            {"name": "+ LLM rewrite + LLM coverage verdict (production)", "mode": "hybrid", "reranker": prod,
             "translate": True, "rewrite": True, "verdict": True, "production": True},
        ]
    return out


def first_relevant_rank(passages: List[Dict], expected: List[List[str]]) -> Optional[int]:
    for rank, p in enumerate(passages, start=1):
        for source, section in expected:
            if p["source"] == source and section.lower() in p["section"].lower():
                return rank
    return None


async def run_variant(store: KnowledgeStore, questions: List[Dict], v: Dict, helper: Optional[LLMHelper]) -> List[Dict]:
    sem = asyncio.Semaphore(6)

    async def one(q: Dict) -> Dict:
        question = q["question"]
        query, also = question, ()
        if v.get("rewrite") and helper:
            async with sem:
                query = await helper.rewrite(question)
            also = (question,)  # the agent's English query is searched together with the customer's own words
        res = store.search_kb(query, product_line=q.get("scope", "general"), k=TOP_N, reranker=v["reranker"],
                              mode=v["mode"], translate=v["translate"], also=also)
        passages = res["policy_passages"]
        covered = bool(res["match_found"])
        if v.get("verdict") and helper:
            async with sem:
                covered = bool(passages) and await helper.verdict(question, passages)
        answerable = q.get("answerable", True)
        rank = first_relevant_rank(passages, q["expected"]) if answerable else None
        return {
            "variant": v["name"], "id": q["id"], "split": q.get("split", "dev"), "lang": q["lang"],
            "scope": q.get("scope", "general"), "answerable": answerable, "question": question,
            "search_query": query if query != question else "", "rank": rank,
            "hit@1": answerable and rank is not None and rank <= 1,
            "hit@3": answerable and rank is not None and rank <= 3,
            "hit@5": answerable and rank is not None and rank <= 5,
            "rr": (1 / rank) if rank else 0.0,
            "top_relevance": res["top_relevance"], "top_coverage": res["top_coverage"],
            "match_found": covered, "abstention_ok": covered == answerable,
            "top1": f"{passages[0]['source']} > {passages[0]['section'][:70]}" if passages else "",
            "top_found_by": passages[0].get("found_by", "") if passages else "",
            "queries": " | ".join(res.get("search_queries", [])),
        }

    return list(await asyncio.gather(*(one(q) for q in questions)))


# ------------------------------------------------------------------ metrics
def _ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> str:
    """95% bootstrap confidence interval of a mean, as 'lo-hi'."""
    if len(values) == 0:
        return ""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return f"{np.percentile(means, 2.5):.2f}-{np.percentile(means, 97.5):.2f}"


def summarise(rows: List[Dict]) -> List[Dict]:
    """One row per (split, variant); split 'all' = dev + held-out test."""
    df = pd.DataFrame(rows)
    out = []
    for split in ["test", "dev", "all"]:
        part = df if split == "all" else df[df["split"] == split]
        for name, g in part.groupby("variant", sort=False):
            ans = g[g["answerable"]]
            indic = ans[ans["lang"] != "en"]
            out.append({
                "split": SPLIT_NAMES[split], "variant": name, "questions": int(len(g)),
                "hit@1": round(ans["hit@1"].mean(), 3), "hit@3": round(ans["hit@3"].mean(), 3),
                "hit@3 95% CI": _ci(ans["hit@3"].astype(float).to_numpy()),
                "hit@5": round(ans["hit@5"].mean(), 3), "MRR@10": round(ans["rr"].mean(), 3),
                "MRR 95% CI": _ci(ans["rr"].astype(float).to_numpy()),
                "hit@3 Tamil/Hindi": round(indic["hit@3"].mean(), 3) if len(indic) else None,
                "abstention accuracy": round(g["abstention_ok"].mean(), 3),
                "abstention 95% CI": _ci(g["abstention_ok"].astype(float).to_numpy()),
                "false 'covered' on unanswerable": int(((~g["answerable"]) & g["match_found"]).sum()),
                "missed answerable": int((g["answerable"] & ~g["match_found"]).sum()),
            })
    return out


def isolated(settings: Settings) -> Settings:
    """Measure the policy retriever alone: learned solutions grow with every ticket and would make the numbers
    depend on how the desk was used, so the evaluation points at an empty learned-solutions file."""
    return settings.with_overrides(learned_file=BASE_DIR / "evaluation" / "runs" / "_no_learned_solutions.jsonl")


def _use_llm(settings: Settings, use_llm: Optional[bool]) -> bool:
    return (not settings.is_offline) if use_llm is None else (use_llm and not settings.is_offline)


def _make_helper(settings: Settings, llm: bool) -> Optional[LLMHelper]:
    """The LLM variants need a working provider; without one (no key) they are skipped, not fatal."""
    if not llm:
        return None
    try:
        return LLMHelper(settings)
    except Exception as exc:
        print(f"[retrieval_eval] LLM variants skipped: {exc}")
        return None


async def evaluate_async(settings: Optional[Settings] = None, write: bool = True,
                         use_llm: Optional[bool] = None) -> Dict:
    settings = isolated(settings or get_settings())
    store = KnowledgeStore(settings)
    questions = load_set()
    helper = _make_helper(settings, _use_llm(settings, use_llm))
    llm = helper is not None
    rows: List[Dict] = []
    try:
        for v in variants(settings, llm):
            rows.extend(await run_variant(store, questions, v, helper))
    finally:
        if helper:
            helper.save()
            await helper.close()
    summary = summarise(rows)
    prod_name = next(v["name"] for v in variants(settings, llm) if v.get("production"))
    production = {split: next(s for s in summary if s["variant"] == prod_name and s["split"] == split)
                  for split in SPLIT_NAMES.values()}
    fingerprint = split_fingerprint(questions)
    frozen = frozen_fingerprint()
    integrity = {"test_split_sha256": fingerprint, "unchanged_since_freeze": bool(frozen) and frozen == fingerprint,
                 "frozen_at": json.loads(SET_FILE.read_text(encoding="utf-8")).get("test_split_frozen_at", ""),
                 "leaked_into_code": leaked_questions(questions), "thresholds": store.thresholds()}
    if write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(REPORT_DIR / "retrieval_metrics.csv", index=False)
        pd.DataFrame(rows).to_csv(REPORT_DIR / "retrieval_cases.csv", index=False)
    return {"summary": summary, "production": production, "rows": rows,
            "embedding": f"{settings.embedding_backend} {settings.embedding_model}".strip(),
            "reranker": settings.reranker, "llm": helper.model if helper else "none (offline)",
            "llm_calls": helper.calls if helper else 0, "questions": len(questions),
            "languages": sorted({q["lang"] for q in questions}), "integrity": integrity}


def evaluate(settings: Optional[Settings] = None, write: bool = True, use_llm: Optional[bool] = None) -> Dict:
    return asyncio.run(evaluate_async(settings, write, use_llm))


# ------------------------------------------------------------------ calibration (dev split only)
# A false 'covered' (the desk answering what the documents do not say) is worse than one more ticket for a human,
# so it costs three needless escalations. Not more: a needless escalation is also a failure the customer sees
# ("where is my order?" sent to a human), and a false 'covered' still has to pass the grounding check, the LLM
# claim verifier and the guardrails' off-topic filter before a reply goes out.
FALSE_COVERED_COST = 3.0


async def calibrate_async(settings: Optional[Settings] = None, use_llm: Optional[bool] = None,
                          write: bool = True, false_covered_cost: float = FALSE_COVERED_COST) -> Dict:
    """Tune KB_MIN_RELEVANCE / KB_MIN_COVERAGE / KB_STRONG_RELEVANCE on the DEV questions for this backend.

    The search does not depend on the thresholds, so every dev question is searched once (through the same
    path as production: with the LLM query rewrite when an LLM is configured) and the thresholds are swept on
    the recorded signals. Objective: lowest cost = missed answerable + ``false_covered_cost`` x false 'covered'
    (the bot answering what the documents do not say is worse than sending a ticket to a human). Several
    settings usually tie; the one in the middle of the tied range is chosen, so a slightly different question
    does not flip the decision. Written to evaluation/calibration/<backend>.json and used automatically while
    KB_MIN_* are 'auto' in .env.
    """
    settings = isolated(settings or get_settings())
    store = KnowledgeStore(settings)
    questions = [q for q in load_set() if q.get("split", "dev") == "dev"]
    helper = _make_helper(settings, _use_llm(settings, use_llm))
    signals = []
    try:
        for q in questions:
            query, also = q["question"], ()
            if helper:
                query, also = await helper.rewrite(q["question"]), (q["question"],)
            res = store.search_kb(query, product_line=q.get("scope", "general"), also=also)
            signals.append((q.get("answerable", True), res["top_relevance"], res["top_coverage"]))
    finally:
        if helper:
            helper.save()
            await helper.close()
    ans = np.array([a for a, _, _ in signals], dtype=bool)
    rel = np.array([r for _, r, _ in signals], dtype=float)
    cov = np.array([c for _, _, c in signals], dtype=float)

    def cuts(values: np.ndarray, lo: float, hi: float) -> List[float]:
        """Candidate thresholds: half-way between neighbouring observed values (every distinct decision), in range."""
        v = np.unique(np.round(values, 4))
        mids = np.concatenate([[v[0] / 2], (v[:-1] + v[1:]) / 2, [v[-1] + 0.01]])
        return sorted({round(float(x), 3) for x in mids if lo <= x <= hi})

    rels = cuts(rel, 0.0, float(np.percentile(rel, 70)))
    covs = cuts(cov, 0.20, 0.60)
    stro = cuts(rel, float(np.percentile(rel, 40)), float(rel.max()) + 0.01)
    rows = []
    for r_th in rels:
        base = rel >= r_th
        for c_th in covs:
            for s_th in stro:
                s_eff = max(s_th, r_th)
                pred = base & ((cov >= c_th) | ((rel >= s_eff) & (cov >= STRONG_MIN_COVERAGE)))
                tp, fn = int((ans & pred).sum()), int((ans & ~pred).sum())
                tn, fp = int((~ans & ~pred).sum()), int((~ans & pred).sum())
                rows.append({"min_relevance": r_th, "min_coverage": c_th, "strong_relevance": s_eff,
                             "cost": fn + false_covered_cost * fp, "accuracy": round((tp + tn) / len(ans), 3),
                             "balanced_accuracy": round(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), 3),
                             "false_covered": fp, "missed_answerable": fn})
    table = pd.DataFrame(rows)
    tied = table[table["cost"] == table["cost"].min()]
    tied = tied[tied["false_covered"] == tied["false_covered"].min()].copy()
    for col in ("min_relevance", "min_coverage", "strong_relevance"):  # distance to the middle of the tied range
        span = (tied[col].max() - tied[col].min()) or 1.0
        tied[f"_{col}"] = (tied[col] - (tied[col].max() + tied[col].min()) / 2).abs() / span
    tied["_mid"] = tied[["_min_relevance", "_min_coverage", "_strong_relevance"]].sum(axis=1)
    best = tied.sort_values("_mid").iloc[0][list(table.columns)].to_dict()
    table = table.sort_values(["cost", "false_covered", "accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    result = {"min_relevance": float(best["min_relevance"]), "min_coverage": float(best["min_coverage"]),
              "strong_relevance": float(best["strong_relevance"]), "strong_min_coverage": STRONG_MIN_COVERAGE,
              "dev_accuracy": float(best["accuracy"]), "dev_balanced_accuracy": float(best["balanced_accuracy"]),
              "false_covered_cost": false_covered_cost, "tied_settings": int(len(tied)),
              "dev_false_covered": int(best["false_covered"]), "dev_missed_answerable": int(best["missed_answerable"]),
              "embedding_backend": settings.embedding_backend, "embedding_model": settings.embedding_model,
              "with_llm_rewrite": bool(helper), "dev_questions": len(questions),
              "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    if write:
        path = calibration_file(settings.embedding_backend, settings.embedding_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        result["file"] = str(path.relative_to(BASE_DIR))
    return {"best": result, "table": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-write", action="store_true", help="print only, do not write reports / calibration")
    parser.add_argument("--calibrate", action="store_true",
                        help="tune the 'covered' thresholds on the dev questions and save them for this backend")
    parser.add_argument("--embedding", help="tfidf | openai | sentence-transformers (default: EMBEDDING_BACKEND)")
    parser.add_argument("--embedding-model", default=None, help="e.g. text-embedding-3-small")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM rewrite / verdict variants")
    parser.add_argument("--false-covered-cost", type=float, default=FALSE_COVERED_COST,
                        help="calibration: cost of a false 'covered' relative to a needless escalation (default 3)")
    args = parser.parse_args()
    overrides = {}
    if args.embedding:
        overrides["embedding_backend"] = args.embedding.lower()
    if args.embedding_model is not None:
        overrides["embedding_model"] = args.embedding_model
    settings = get_settings(**overrides)
    use_llm = False if args.no_llm else None
    if args.calibrate:
        cal = asyncio.run(calibrate_async(settings, use_llm, write=not args.no_write,
                                          false_covered_cost=args.false_covered_cost))
        print("'Covered' thresholds on the dev split (best first):\n")
        print(cal["table"].head(8).to_string(index=False))
        b = cal["best"]
        print(f"\n-> min_relevance={b['min_relevance']}  min_coverage={b['min_coverage']}  "
              f"strong_relevance={b['strong_relevance']}  (dev accuracy {b['dev_accuracy']}, false covered "
              f"{b['dev_false_covered']}, missed answerable {b['dev_missed_answerable']}; middle of "
              f"{b['tied_settings']} equally good settings)")
        if b.get("file"):
            print(f"Saved to {b['file']} - used automatically while KB_MIN_RELEVANCE / KB_MIN_COVERAGE / "
                  "KB_STRONG_RELEVANCE are 'auto' in .env.")
        return
    rep = evaluate(settings, write=not args.no_write, use_llm=use_llm)
    integ = rep["integrity"]
    print(f"{rep['questions']} labelled questions ({', '.join(rep['languages'])}) · embeddings: {rep['embedding']}"
          f" · reranker: {rep['reranker']} · LLM: {rep['llm']} ({rep['llm_calls']} new calls)")
    print(f"Thresholds: {integ['thresholds']}")
    print(f"Held-out split fingerprint {integ['test_split_sha256'][:16]}… "
          f"{'unchanged since ' + integ['frozen_at'] if integ['unchanged_since_freeze'] else 'CHANGED / not frozen'}"
          f" · held-out questions found in code: {integ['leaked_into_code'] or 'none'}\n")
    table = pd.DataFrame(rep["summary"])
    cols = ["variant", "hit@1", "hit@3", "hit@3 95% CI", "hit@5", "MRR@10", "hit@3 Tamil/Hindi",
            "abstention accuracy", "false 'covered' on unanswerable", "missed answerable"]
    for split, part in table.groupby("split", sort=False):
        print(f"--- {split}")
        print(part[cols].to_string(index=False))
        print()
    prod = rep["production"]["all"]["variant"]
    misses = [r for r in rep["rows"] if r["variant"] == prod and ((r["answerable"] and not r["hit@3"])
                                                                  or not r["abstention_ok"])]
    if misses:
        print(f"Misses of the production path ({prod}): not in top 3, or wrong covered / not-covered decision:")
        for r in misses:
            print(f"  {r['id']:6} [{r['split']}] rank={r['rank']} covered={r['match_found']} top1={r['top1']}")
    if not args.no_write:
        print(f"\nSaved: {REPORT_DIR / 'retrieval_metrics.csv'} and retrieval_cases.csv")


if __name__ == "__main__":
    main()
