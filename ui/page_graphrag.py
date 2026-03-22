"""
🔍 GraphRAG page — Graph-Retrieval Augmented Generation.
"""

from __future__ import annotations

import streamlit as st

from ui.components import render_retrieval_detail as _render_retrieval_detail


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_pipeline(
    strategy: str,
    retrieval_methods: list,
    format_keys: list,
    enable_enhancement: bool,
):
    """Create (or reuse) a cached GraphRAGPipeline in session_state."""
    cache_key = f"graphrag_pipeline_{strategy}_{'_'.join(retrieval_methods)}"
    if cache_key not in st.session_state:
        from src.graph_loader.neo4j_client import Neo4jClient
        from src.pipeline.pipeline import GraphRAGPipeline

        client = Neo4jClient()
        client.ping()  # raises if Neo4j is unreachable
        st.session_state[cache_key] = GraphRAGPipeline(
            client,
            retrieval_methods=retrieval_methods,
            format_keys=format_keys,
            generation_strategy=strategy,
            enable_query_enhancement=enable_enhancement,
        )
    return st.session_state[cache_key]


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("🔍 GraphRAG — Hỏi đáp về Chèo")
    st.caption("Graph-Retrieval Augmented Generation: kết hợp knowledge graph + LLM")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Cài đặt pipeline")

    from src.constants.constant import FormatKey, GenerationStrategy, RetrievalMethod

    strategy = st.sidebar.selectbox(
        "Generation strategy",
        options=[GenerationStrategy.PRE, GenerationStrategy.MID, GenerationStrategy.POST],
        index=1,
        format_func=lambda s: {
            "pre":  "Pre (context first)",
            "mid":  "Mid (structured guidance)",
            "post": "Post (verify & refine)",
        }[s],
    )

    retrieval_methods = st.sidebar.multiselect(
        "Retrieval methods",
        options=RetrievalMethod.ALL,
        default=RetrievalMethod.DEFAULT,
    )

    format_keys = st.sidebar.multiselect(
        "Context formats",
        options=[
            FormatKey.NATURAL_LANGUAGE,
            FormatKey.ADJACENCY_TABLE,
            FormatKey.CODE_LIKE,
            FormatKey.NODE_SEQUENCE,
            FormatKey.EMBEDDING_TEXT,
        ],
        default=[FormatKey.NATURAL_LANGUAGE, FormatKey.CODE_LIKE],
    )

    enable_enhancement = st.sidebar.toggle(
        "Query enhancement",
        value=True,
        help="Expand / decompose complex queries before retrieval",
    )

    show_detail = st.sidebar.toggle(
        "Hiện chi tiết retrieval",
        value=True,
        help="Hiển thị nodes, triplets, paths, context đã thu thập được",
    )

    if st.sidebar.button("🗑️ Xóa lịch sử"):
        st.session_state.graphrag_history = []
        st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    if "graphrag_history" not in st.session_state:
        st.session_state.graphrag_history = []

    for item in st.session_state.graphrag_history:
        with st.chat_message("user"):
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if show_detail:
                with st.expander("📊 Chi tiết retrieval", expanded=False):
                    _render_retrieval_detail(item)

    # ── Guards ────────────────────────────────────────────────────────────────
    if not retrieval_methods:
        st.warning("Chọn ít nhất một retrieval method ở sidebar.")
        return
    if not format_keys:
        st.warning("Chọn ít nhất một context format ở sidebar.")
        return

    # ── Input ─────────────────────────────────────────────────────────────────
    if query := st.chat_input("Hỏi về Chèo (vd: Thúy Kiều là ai?)..."):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất graph và sinh câu trả lời..."):
                try:
                    pipeline = _get_pipeline(
                        strategy=strategy,
                        retrieval_methods=retrieval_methods,
                        format_keys=format_keys,
                        enable_enhancement=enable_enhancement,
                    )
                    result = pipeline.run(query)

                    if result.success:
                        st.markdown(result.answer)
                    else:
                        err = (
                            result.generation.error
                            or result.retrieval.error
                            or "Unknown error"
                        )
                        st.error(f"❌ Pipeline thất bại: {err}")

                    # ── Build history item with full retrieval data ─────────
                    history_item = {
                        "query":               query,
                        "answer":              result.answer,
                        "num_nodes":           result.retrieval.num_nodes,
                        "num_triplets":        result.retrieval.num_triplets,
                        "num_paths":           result.retrieval.num_paths,
                        "retrieval_time":      result.retrieval.retrieval_time,
                        "gen_time":            result.generation.generation_time,
                        "total_time":          result.total_time,
                        # Detailed data
                        "processed_query":     dict(result.retrieval.processed_query),
                        "entities":            dict(result.retrieval.entities),
                        "nodes":               list(result.retrieval.graph_data.get("nodes", [])),
                        "triplets":            list(result.retrieval.graph_data.get("triplets", [])),
                        "paths":               list(result.retrieval.graph_data.get("paths", [])),
                        "formatted_contexts":  dict(result.retrieval.formatted_contexts),
                    }

                    if show_detail:
                        with st.expander("📊 Chi tiết retrieval", expanded=True):
                            _render_retrieval_detail(history_item)

                    st.session_state.graphrag_history.append(history_item)

                except Exception as exc:
                    st.error(f"❌ Lỗi: {exc}")
                    st.info("Đảm bảo Neo4j đang chạy và đã load dữ liệu ontology.")
