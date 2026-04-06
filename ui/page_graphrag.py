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

_EXAMPLE_QUESTIONS = [
    "Vở chèo Quan Âm Thị Kính có những nhân vật chính nào?",
    "Diễn viên Thanh Ngoan đã thủ vai những nhân vật gì?",
    "Nhân vật Thị Mầu xuất hiện trong những trích đoạn nào?",
    "Liệt kê các vở chèo trong hệ thống và nhân vật nữ chính.",
    "Trích đoạn Thị Mầu lên chùa có những phiên bản nào và ai diễn?",
]


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
        from src.utils.history_store import HistoryStore
        HistoryStore.clear(page_filter="graphrag")
        st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    if "graphrag_history" not in st.session_state:
        st.session_state.graphrag_history = []
    if "graphrag_prefill" not in st.session_state:
        st.session_state.graphrag_prefill = ""

    # ── Restore from persistent storage on first load ──────────────────────
    if not st.session_state.graphrag_history:
        try:
            from src.utils.history_store import HistoryStore
            for e in reversed(HistoryStore.load(page_filter="graphrag")):
                st.session_state.graphrag_history.append({
                    "query":              e["query"],
                    "answer":             e["answer"],
                    "num_nodes":          e["metadata"].get("num_nodes", 0),
                    "num_triplets":       e["metadata"].get("num_triplets", 0),
                    "num_paths":          0,
                    "retrieval_time":     e["metadata"].get("retrieval_time", 0.0),
                    "gen_time":           0.0,
                    "total_time":         e["metadata"].get("total_time", 0.0),
                    "processed_query":    {},
                    "entities":           {},
                    "nodes":              [],
                    "triplets":           [],
                    "paths":              [],
                    "subgraph":           {},
                    "formatted_contexts": {},
                    "_persisted":         True,  # đánh dấu từ file, không có graph detail
                })
        except Exception:
            pass

    # ── Onboarding (chỉ hiển thị khi chưa có lịch sử) ────────────────────────
    if not st.session_state.graphrag_history:
        st.info(
            "💡 **Hệ thống GraphRAG** truy xuất thông tin từ Knowledge Graph về "
            "nghệ thuật Chèo Việt Nam, sau đó dùng LLM để tổng hợp câu trả lời.\n\n"
            "Hãy thử đặt câu hỏi về **nhân vật**, **vở kịch**, **diễn viên** hoặc **trích đoạn** Chèo:"
        )
        cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for col, q in zip(cols, _EXAMPLE_QUESTIONS):
            if col.button(q, use_container_width=True, key=f"eg_{q[:20]}"):
                st.session_state.graphrag_prefill = q
                st.rerun()

    for h_idx, item in enumerate(st.session_state.graphrag_history):
        with st.chat_message("user"):
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if show_detail:
                with st.expander("📊 Chi tiết retrieval", expanded=False):
                    _render_retrieval_detail(item, key_prefix=f"gr_h{h_idx}_")

    # ── Guards ────────────────────────────────────────────────────────────────
    if not retrieval_methods:
        st.warning("Chọn ít nhất một retrieval method ở sidebar.")
        return
    if not format_keys:
        st.warning("Chọn ít nhất một context format ở sidebar.")
        return

    # ── Input ─────────────────────────────────────────────────────────────────
    prefill = st.session_state.pop("graphrag_prefill", "")
    if query := (st.chat_input("Hỏi về Chèo (vd: Thị Mầu là ai?)...") or prefill or "") or None:
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
                        "subgraph":            dict(result.retrieval.graph_data.get("subgraph", {})),
                        "formatted_contexts":  dict(result.retrieval.formatted_contexts),
                    }

                    if show_detail:
                        with st.expander("📊 Chi tiết retrieval", expanded=True):
                            _render_retrieval_detail(history_item, key_prefix=f"gr_new_{len(st.session_state.graphrag_history)}_")

                    st.session_state.graphrag_history.append(history_item)

                    # ── Persist to history file ────────────────────────────
                    if result.success:
                        from src.utils.history_store import HistoryStore
                        HistoryStore.append(
                            page="graphrag",
                            query=query,
                            answer=result.answer,
                            metadata={
                                "num_nodes":      result.retrieval.num_nodes,
                                "num_triplets":   result.retrieval.num_triplets,
                                "retrieval_time": round(result.retrieval.retrieval_time, 2),
                                "total_time":     round(result.total_time, 2),
                                "strategy":       strategy,
                            },
                        )

                except Exception as exc:
                    st.error(f"❌ Lỗi: {exc}")
                    st.info("Đảm bảo Neo4j đang chạy và đã load dữ liệu ontology.")
