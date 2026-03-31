"""
📚 RAG page — Traditional Vector-Search RAG.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Vector store pickle lives at project-root/data/vector_store.pkl
_RAG_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "vector_store.pkl"


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_rag_pipeline(top_k: int):
    """Create (or reuse) a cached VectorRAGPipeline in session_state."""
    key = f"rag_pipeline_{top_k}"
    if key not in st.session_state:
        from src.rag.pipeline import VectorRAGPipeline
        st.session_state[key] = VectorRAGPipeline(
            store_path=_RAG_STORE_PATH, top_k=top_k
        )
    return st.session_state[key]


# ── Render ────────────────────────────────────────────────────────────────────

_EXAMPLE_QUESTIONS = [
    "Chèo là gì? Hãy giới thiệu về nghệ thuật Chèo Việt Nam.",
    "Vở chèo Quan Âm Thị Kính kể về chủ đề gì?",
    "Nhân vật Thị Mầu trong Chèo có đặc điểm gì?",
    "Trích đoạn Súy Vân giả dại nói về điều gì?",
]


def render() -> None:
    st.title("📚 RAG — Truyền thống Vector Search")
    st.caption(
        "Traditional Retrieval-Augmented Generation: "
        "nhúng câu hỏi → tìm kiếm cosine → sinh câu trả lời."
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Cài đặt RAG")

    top_k = st.sidebar.slider(
        "Số chunks truy xuất (top-k)",
        min_value=1, max_value=20, value=5, step=1,
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🗄️ Vector Store")

    store_exists = _RAG_STORE_PATH.exists()
    if store_exists:
        size_mb = _RAG_STORE_PATH.stat().st_size / 1024 / 1024
        st.sidebar.success(f"✅ Store sẵn sàng ({size_mb:.2f} MB)")
    else:
        st.sidebar.warning("⚠️ Chưa có vector store")

    if st.sidebar.button(
        "🔨 Build / Rebuild Vector Store",
        help="Kéo dữ liệu từ Neo4j, nhúng và lưu pickle",
        use_container_width=True,
    ):
        # Invalidate cached pipelines so they reload the new store
        for k in list(st.session_state.keys()):
            if k.startswith("rag_pipeline_"):
                del st.session_state[k]

        with st.sidebar.status("Đang build vector store …", expanded=True) as s:
            try:
                from src.rag.prepare_data import build_vector_store

                store = build_vector_store(output_path=_RAG_STORE_PATH)
                s.update(
                    label=f"✅ Xong! {len(store)} chunks",
                    state="complete",
                    expanded=False,
                )
                st.rerun()
            except Exception as exc:
                s.update(label=f"❌ {exc}", state="error")

    if st.sidebar.button("🗑️ Xóa lịch sử"):
        st.session_state.rag_history = []
        from src.utils.history_store import HistoryStore
        HistoryStore.clear(page_filter="rag")
        st.rerun()

    # ── Guard: store missing ──────────────────────────────────────────────────
    if not store_exists:
        st.info(
            "Vector store chưa được xây dựng.  \n"
            "Nhấn **Build / Rebuild Vector Store** ở sidebar (cần Neo4j đang chạy)."
        )
        return

    # ── History ───────────────────────────────────────────────────────────────
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    if "rag_prefill" not in st.session_state:
        st.session_state.rag_prefill = ""

    # ── Restore from persistent storage on first load ──────────────────────
    if not st.session_state.rag_history:
        try:
            from src.utils.history_store import HistoryStore
            for e in reversed(HistoryStore.load(page_filter="rag")):
                st.session_state.rag_history.append({
                    "query":          e["query"],
                    "answer":         e["answer"],
                    "num_chunks":     e["metadata"].get("num_chunks", 0),
                    "context_length": 0,
                    "retrieval_time": e["metadata"].get("retrieval_time", 0.0),
                    "gen_time":       e["metadata"].get("gen_time", 0.0),
                })
        except Exception:
            pass

    # ── Onboarding (chỉ hiển thị khi chưa có lịch sử) ────────────────────────
    if not st.session_state.rag_history:
        st.info(
            "💡 **Hệ thống RAG** tìm kiếm các đoạn văn bản liên quan nhất bao tiết kho vector "
            "rồi dùng LLM để trả lời. Phù hợp cho câu hỏi mô tả tổng quan về Chèo.\n\n"
            "Thử đặt câu hỏi:"
        )
        cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for col, q in zip(cols, _EXAMPLE_QUESTIONS):
            if col.button(q, use_container_width=True, key=f"rag_eg_{q[:20]}"):
                st.session_state.rag_prefill = q
                st.rerun()

    for item in st.session_state.rag_history:
        with st.chat_message("user"):
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander("📊 Chi tiết"):
                col1, col2 = st.columns(2)
                col1.metric("Chunks truy xuất", item["num_chunks"])
                col2.metric("Độ dài context",   item["context_length"])
                st.caption(
                    f"Retrieval: {item['retrieval_time']:.2f}s | "
                    f"Generation: {item['gen_time']:.2f}s"
                )

    # ── Input ─────────────────────────────────────────────────────────────────
    prefill = st.session_state.pop("rag_prefill", "")
    if query := (st.chat_input("Hỏi về Chèo (RAG)...") or prefill or "") or None:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và sinh câu trả lời..."):
                try:
                    pipeline = _get_rag_pipeline(top_k)
                    result   = pipeline.run(query)

                    if result.generation.error:
                        st.error(f"❌ {result.generation.error}")
                    else:
                        st.markdown(result.answer)

                    with st.expander("📊 Chi tiết"):
                        col1, col2 = st.columns(2)
                        col1.metric("Chunks truy xuất", result.retrieval.num_nodes)
                        col2.metric(
                            "Độ dài context",
                            len(result.retrieval.formatted_contexts.get(
                                "natural_language", ""
                            )),
                        )
                        st.caption(
                            f"Retrieval: {result.retrieval.retrieval_time:.2f}s | "
                            f"Generation: {result.generation.generation_time:.2f}s"
                        )
                        if result.retrieval.key_facts:
                            st.markdown("**Key facts:**")
                            st.markdown(result.retrieval.key_facts)

                    st.session_state.rag_history.append({
                        "query":          query,
                        "answer":         result.answer,
                        "num_chunks":     result.retrieval.num_nodes,
                        "context_length": result.generation.context_length,
                        "retrieval_time": result.retrieval.retrieval_time,
                        "gen_time":       result.generation.generation_time,
                    })

                    # ── Persist to history file ────────────────────────────
                    if not result.generation.error:
                        from src.utils.history_store import HistoryStore
                        HistoryStore.append(
                            page="rag",
                            query=query,
                            answer=result.answer,
                            metadata={
                                "num_chunks":     result.retrieval.num_nodes,
                                "retrieval_time": round(result.retrieval.retrieval_time, 2),
                                "gen_time":       round(result.generation.generation_time, 2),
                            },
                        )

                except FileNotFoundError:
                    st.error(
                        "Vector store không tìm thấy. "
                        "Nhấn Build ở sidebar để tạo lại."
                    )
                except Exception as exc:
                    st.error(f"❌ Lỗi: {exc}")
