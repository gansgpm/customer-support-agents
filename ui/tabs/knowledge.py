"""Knowledge Base tab: documents, index status, retrieval tester and learned solutions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.style import esc


def render(app) -> None:
    settings, store, catalog = app.settings, app.store, app.catalog
    info = store.index_info()
    i1, i2, i3, i4, i5 = st.columns([1, 1, 1.4, 1.4, 1])
    i1.metric("Documents", info["documents"])
    i2.metric("Chunks", info["chunks"], help="One chunk per policy section (a complete question + answer)")
    i3.metric("Index", info["origin"], help=f"Saved in {info['folder']} with a manifest of every document's "
              "SHA-256, the embedding model and the chunker; reloaded when nothing changed, rebuilt otherwise.")
    i4.metric("Embeddings", info["embedding"] or "tfidf", help=f"{info['dimensions']} dimensions · "
              f"{info['vector_store']} · chunker {info['chunker']}")
    if i5.button("🔄 Rebuild index", help="Re-chunk and re-embed every document now"):
        store.rebuild_index()
        info = store.index_info()
        st.toast(f"Index rebuilt: {info['chunks']} chunks")
    th = store.thresholds()
    st.caption(f"Built {info['built_at']} · reranker `{settings.reranker}` · candidates {settings.kb_candidates} "
               f"→ top {settings.kb_top_k} · 'covered' needs relevance ≥ {th['min_relevance']} and key-word "
               f"coverage ≥ {th['min_coverage']} (or relevance ≥ {th['strong_relevance']}) · thresholds "
               f"{th['source']} - recalibrate with: python -m evaluation.retrieval_eval --calibrate")
    kb1, kb2 = st.columns([1, 1], gap="large")
    with kb1:
        st.markdown("#### Company documents (policy RAG, scoped by product line)")
        for scope, paths in sorted(store.documents_by_scope().items(), key=lambda kv: kv[0] != "general"):
            st.markdown(f"**{catalog.name(scope)}**")
            for path in paths:
                with st.expander(f"📄 {path.name}"):
                    st.markdown(path.read_text(encoding="utf-8"))
        up = st.file_uploader("Add a general policy document (.md / .txt)", type=["md", "txt"],
                              help="Product-line guides go in knowledge_base/product_lines or services and are "
                                   "listed under the line's documents in catalog.json.")
        if up is not None and st.button("Add to knowledge base"):
            target = Path(settings.kb_dir) / "general" / (Path(up.name).stem + ".md")
            target.write_bytes(up.getvalue())
            st.success(f"Saved {target.name}; the index rebuilds automatically on the next search.")
    with kb2:
        st.markdown("#### Test retrieval")
        q = st.text_input("Ask the knowledge base (any language)", value="how long does a refund take to my card",
                          help="Try Tamil or Hindi too, e.g. பணம் திரும்ப எப்போது வரும்?")
        scope_label = st.selectbox("Scope", ["All documents"] + [l.name for l in catalog.lines], key="kb_scope")
        scope_id = "" if scope_label == "All documents" else catalog.normalize(scope_label)
        if q:
            res = store.search_kb(q, product_line=scope_id)
            st.markdown(f"{'✅ covered' if res['match_found'] else '❌ not covered'} · relevance "
                        f"{res['top_relevance']} · coverage {res['top_coverage']} · {res['retriever']}")
            if len(res.get("search_queries", [])) > 1:
                st.caption(f"Language: {res['query_language']} · searched as: "
                           + " · ".join(f"“{x[:70]}”" for x in res["search_queries"]))
            for p in res["policy_passages"]:
                ranks = p.get("ranks") or {}
                st.markdown(f"**{esc(p['source'])}** <span class='chip'>{esc(catalog.name(p.get('scope', 'general')))}"
                            f"</span> › {esc(p['section'])} <span class='small-muted'>(found by {p.get('found_by')} · "
                            f"BM25 #{ranks.get('bm25') or '-'} · vector #{ranks.get('vector') or '-'} · fusion "
                            f"#{ranks.get('fusion')} · rerank {p.get('rerank_score')} · rel {p['relevance']} · cov "
                            f"{p['coverage']})</span>", unsafe_allow_html=True)
                st.caption(p["text"][:400])
            for p in res["resolved_ticket_matches"]:
                st.markdown(f"🧠 learned **{esc(p['source'])}** (rel {p['relevance']})")
        st.markdown("#### Learned solutions (fed back by the Log Agent)")
        learned = store.learned_entries()
        if learned:
            ldf = pd.DataFrame(learned)
            if "product_line" not in ldf.columns:
                ldf["product_line"] = "general"
            st.dataframe(ldf[["ticket_id", "product_line", "category", "approved_by", "confidence", "reuse_count",
                              "sources", "question"]].iloc[::-1], hide_index=True, width="stretch")
        else:
            st.caption("Empty — resolved tickets with confidence ≥ threshold and human-approved replies appear here.")
