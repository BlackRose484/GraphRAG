"""
🔍 GraphRAG page — Graph-Retrieval Augmented Generation.
"""

from __future__ import annotations

import streamlit as st


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
            with st.expander("📊 Chi tiết retrieval"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Nodes",    item["num_nodes"])
                col2.metric("Triplets", item["num_triplets"])
                col3.metric("Paths",    item["num_paths"])
                st.caption(
                    f"Retrieval: {item['retrieval_time']:.2f}s | "
                    f"Generation: {item['gen_time']:.2f}s | "
                    f"Total: {item['total_time']:.2f}s"
                )

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

                    with st.expander("📊 Chi tiết retrieval"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Nodes",    result.retrieval.num_nodes)
                        c2.metric("Triplets", result.retrieval.num_triplets)
                        c3.metric("Paths",    result.retrieval.num_paths)
                        st.caption(
                            f"Retrieval: {result.retrieval.retrieval_time:.2f}s | "
                            f"Generation: {result.generation.generation_time:.2f}s | "
                            f"Total: {result.total_time:.2f}s"
                        )

                    st.session_state.graphrag_history.append({
                        "query":          query,
                        "answer":         result.answer,
                        "num_nodes":      result.retrieval.num_nodes,
                        "num_triplets":   result.retrieval.num_triplets,
                        "num_paths":      result.retrieval.num_paths,
                        "retrieval_time": result.retrieval.retrieval_time,
                        "gen_time":       result.generation.generation_time,
                        "total_time":     result.total_time,
                    })

                except Exception as exc:
                    st.error(f"❌ Lỗi: {exc}")
                    st.info("Đảm bảo Neo4j đang chạy và đã load dữ liệu ontology.")
