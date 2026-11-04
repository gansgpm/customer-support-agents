"""Evaluation tab: rubric estimate, node scores, retrieval metrics and test cases."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui.style import chips


def render(app) -> None:
    settings = app.settings
    report_path = settings.base_dir / "evaluation" / "reports" / "latest.json"
    st.markdown("#### Node-by-node evaluation against the assignment rubric")
    st.caption("Run from a terminal: `python -m evaluation.evaluate` (uses the provider in .env) or "
               "`python -m evaluation.evaluate --provider offline`.")
    if not report_path.exists():
        st.info("No evaluation report yet.")
    else:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        meta = rep["meta"]
        st.markdown(chips([f"Model: {meta['model']}", f"Run: {meta['run_at']}", f"Cases: {meta['cases_run']}",
                           f"Orchestration: {meta['orchestration']}"]), unsafe_allow_html=True)
        r = rep["rubric"]
        if meta["model"].startswith("Offline"):
            st.warning("This report was produced with the offline demo model, whose rules were written alongside these "
                       "test cases — it proves the pipeline works end to end, not LLM quality. Re-run "
                       "`python -m evaluation.evaluate` with your LLM provider before submitting.")
        st.metric("Automated rubric estimate", f"{r['total']:.0f} / 100")
        st.dataframe(pd.DataFrame(r["areas"]), hide_index=True, width="stretch")
        render_retrieval(settings, rep)
        f = rep.get("faithfulness")
        if f:
            st.markdown("##### Answer faithfulness")
            a, b, c, d = st.columns(4)
            a.metric("Grounded replies", f"{(f['grounded_share'] or 0):.0%}",
                     help="Every figure in the reply is in the retrieved passages and the citations are valid "
                          "(the same check runs on every live ticket)")
            b.metric("Lexical claim support", f"{(f['lexical_support'] or 0):.0%}")
            c.metric("LLM judge: faithfulness", f["judge_faithfulness_avg"] or "-", help=f["judge"])
            d.metric("LLM judge: relevance", f["judge_relevance_avg"] or "-", help=f["judge"])
        st.markdown("##### Node scores")
        st.dataframe(pd.DataFrame(rep["nodes"]), hide_index=True, width="stretch")
        st.markdown("##### Test cases")
        st.dataframe(pd.DataFrame(rep["cases"]).astype(str), hide_index=True, width="stretch")
        md = settings.base_dir / "evaluation" / "reports" / "EVALUATION.md"
        if md.exists():
            with st.expander("Full report (EVALUATION.md)"):
                st.markdown(md.read_text(encoding="utf-8"))


def render_retrieval(settings, rep) -> None:
    """Ground-truth retrieval metrics (evaluation/reports/retrieval_metrics.csv)."""
    path = settings.base_dir / "evaluation" / "reports" / "retrieval_metrics.csv"
    st.markdown("##### Retrieval against hand-labelled ground truth")
    if not path.exists():
        st.caption("Run `python -m evaluation.retrieval_eval` to measure hit-rate@k, MRR and abstention.")
        return
    df = pd.read_csv(path)
    rv = rep.get("retrieval", {})
    integ = rv.get("integrity") or {}
    st.caption(f"{rv.get('questions', '')} questions ({', '.join(rv.get('languages', []))}) with the passage that "
               "answers each one, plus questions the documents do not cover. Dev questions were used for tuning; the "
               "held-out questions are never tuned on — that is the honest number.")
    if integ:
        th = integ.get("thresholds") or {}
        ok = integ.get("unchanged_since_freeze")
        leaked = integ.get("leaked_into_code") or []
        st.markdown(
            f"<span class='chip {'ok' if ok else 'bad'}'>held-out split "
            f"{'unchanged since ' + integ.get('frozen_at', '') if ok else 'CHANGED since it was frozen'}</span> "
            f"<span class='chip {'bad' if leaked else 'ok'}'>held-out questions in code: "
            f"{', '.join(leaked) if leaked else 'none'}</span> "
            f"<span class='chip'>thresholds: {th.get('source', '')}</span>", unsafe_allow_html=True)
    held = df[df["split"] == "held-out test"]
    if not held.empty:
        name = rv.get("production_variant", "")
        prod = held[held["variant"] == name] if name else held.iloc[0:0]
        if prod.empty:  # older report: the production row is the last reranked variant without "no translation"
            prod = held[held["variant"].str.contains("reranker|production") & ~held["variant"].str.contains("no translation")]
            prod = prod.tail(1)
        if not prod.empty:
            row = prod.iloc[0]
            st.caption(f"Production path: **{row['variant']}**")
            m = st.columns(5)
            m[0].metric("hit@1 (held-out)", f"{row['hit@1']:.0%}")
            m[1].metric("hit@3 (held-out)", f"{row['hit@3']:.0%}", help=f"95% CI {row.get('hit@3 95% CI', '')}")
            m[2].metric("MRR@10 (held-out)", f"{row['MRR@10']:.2f}")
            m[3].metric("Tamil/Hindi hit@3", f"{row['hit@3 Tamil/Hindi']:.0%}")
            m[4].metric("Abstention accuracy", f"{row['abstention accuracy']:.0%}",
                        help=f"95% CI {row.get('abstention 95% CI', '')}")
        st.bar_chart(held.set_index("variant")[["hit@1", "hit@3", "MRR@10"]], horizontal=True, height=320,
                     stack=False)
    split = st.radio("Split", ["held-out test", "dev (tuning)", "all"], horizontal=True, key="ret_split")
    st.dataframe(df[df["split"] == split].drop(columns=["split"]), hide_index=True, width="stretch")
