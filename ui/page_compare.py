"""
⚖️ Compare page — hỏi 1 câu, 3 hệ thống trả lời song song.

Gửi cùng 1 câu hỏi tới GraphRAG, RAG, và LLM đồng thời,
hiển thị kết quả 3 cột cạnh nhau để người dùng so sánh.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from ui.components import render_retrieval_detail as _render_retrieval_detail

# ── Constants ─────────────────────────────────────────────────────────────────

_RAG_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "vector_store.pkl"

_EXAMPLE_QUESTIONS = [
    "Mô tả đặc điểm nhân vật Súy Vân.",
    "Liệt kê nhân vật trong vở Quan Âm Thị Kính.",
    "An Chinh đóng vai gì trong vở Kim Nham?",
    "Diễn viên nào thủ vai trong nhiều trích đoạn nhất?",
]

# System prompt for LLM-only mode
_CHAT_SYSTEM_PROMPT = (
    "Bạn là trợ lý chuyên về nghệ thuật Chèo Việt Nam. "
    "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác."
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class _SystemResult:
    """Result from one system."""
    answer: str
    elapsed: float
    error: Optional[str] = None
    metadata: dict[str, Any] | None = None
    # GraphRAG-specific retrieval detail (for expander)
    retrieval_detail: dict[str, Any] | None = None


# ── Runner functions (thread-safe) ────────────────────────────────────────────

def _run_graphrag(query: str, disable_query_enhancement: bool = False) -> _SystemResult:
    """Run GraphRAG pipeline.

    Args:
        query: User question.
        disable_query_enhancement: If True, skip the LLM-based query expand/
            decompose step before retrieval (used for user-study pages where
            we want to evaluate the raw pipeline without extra pre-processing).
    """
    t0 = time.time()
    try:
        from src.graph_loader.neo4j_client import Neo4jClient
        from src.pipeline.pipeline import GraphRAGPipeline

        client = Neo4jClient()
        client.ping()
        pipeline = GraphRAGPipeline(
            client,
            enable_query_enhancement=not disable_query_enhancement,
        )
        result = pipeline.run(query)

        detail = {
            "query":              query,
            "answer":             result.answer,
            "num_nodes":          result.retrieval.num_nodes,
            "num_triplets":       result.retrieval.num_triplets,
            "num_paths":          result.retrieval.num_paths,
            "retrieval_time":     result.retrieval.retrieval_time,
            "gen_time":           result.generation.generation_time,
            "total_time":         result.total_time,
            "processed_query":    dict(result.retrieval.processed_query),
            "entities":           dict(result.retrieval.entities),
            "nodes":              list(result.retrieval.graph_data.get("nodes", [])),
            "triplets":           list(result.retrieval.graph_data.get("triplets", [])),
            "paths":              list(result.retrieval.graph_data.get("paths", [])),
            "subgraph":           dict(result.retrieval.graph_data.get("subgraph", {})),
            "formatted_contexts": dict(result.retrieval.formatted_contexts),
        }

        return _SystemResult(
            answer=result.answer,
            elapsed=time.time() - t0,
            metadata={
                "num_nodes":      result.retrieval.num_nodes,
                "num_triplets":   result.retrieval.num_triplets,
                "retrieval_time": round(result.retrieval.retrieval_time, 2),
                "total_time":     round(result.total_time, 2),
            },
            retrieval_detail=detail,
        )
    except Exception as exc:
        return _SystemResult(
            answer="", elapsed=time.time() - t0, error=str(exc)
        )


def _run_rag(query: str) -> _SystemResult:
    """Run traditional RAG pipeline."""
    t0 = time.time()
    try:
        from src.rag.pipeline import VectorRAGPipeline

        pipeline = VectorRAGPipeline(store_path=_RAG_STORE_PATH, top_k=10)
        result = pipeline.run(query)

        if result.generation.error:
            return _SystemResult(
                answer="", elapsed=time.time() - t0,
                error=result.generation.error,
            )
        return _SystemResult(
            answer=result.answer,
            elapsed=time.time() - t0,
            metadata={
                "num_chunks":     result.retrieval.num_nodes,
                "retrieval_time": round(result.retrieval.retrieval_time, 2),
            },
        )
    except Exception as exc:
        return _SystemResult(
            answer="", elapsed=time.time() - t0, error=str(exc)
        )


def _run_chat(query: str) -> _SystemResult:
    """Run LLM-only (no retrieval)."""
    t0 = time.time()
    try:
        from src.core.base import BaseModel

        class _ChatModel(BaseModel):
            pass

        model = _ChatModel()
        prompt = (
            f"{_CHAT_SYSTEM_PROMPT}\n\n"
            f"Người dùng: {query}\n"
            f"Trợ lý:"
        )
        answer = model.safe_generate(prompt)
        return _SystemResult(answer=answer, elapsed=time.time() - t0)
    except Exception as exc:
        return _SystemResult(
            answer="", elapsed=time.time() - t0, error=str(exc)
        )


# ── Parallel executor ─────────────────────────────────────────────────────────

def _run_all(
    query: str,
    disable_query_enhancement: bool = False,
) -> dict[str, _SystemResult]:
    """Run all 3 systems in parallel, return results keyed by system name.

    Args:
        query: User question.
        disable_query_enhancement: Forwarded to GraphRAG; skips query expand/
            decompose. RAG and Chat are unaffected (they don't enhance).
    """
    runners = {
        "graphrag": lambda q: _run_graphrag(q, disable_query_enhancement),
        "rag":      _run_rag,
        "chat":     _run_chat,
    }
    results: dict[str, _SystemResult] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {pool.submit(fn, query): name for name, fn in runners.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = _SystemResult(answer="", elapsed=0, error=str(exc))

    return results


# ── UI Column Rendering ──────────────────────────────────────────────────────

_COLUMN_META = {
    "graphrag": {"title": "🔍 GraphRAG",   "color": "#c05000"},
    "rag":      {"title": "📚 RAG",         "color": "#1565c0"},
    "chat":     {"title": "💬 LLM thuần",   "color": "#1b6b3a"},
}


def _render_column(col, name: str, result: _SystemResult, key_prefix: str = "") -> None:
    """Render one system's result inside a Streamlit column."""
    meta = _COLUMN_META[name]

    # Header
    col.markdown(
        f"<div style='border-bottom:3px solid {meta['color']};padding-bottom:4px;"
        f"margin-bottom:8px;font-weight:700;font-size:1.1rem'>"
        f"{meta['title']}</div>",
        unsafe_allow_html=True,
    )

    if result.error:
        col.error(f"❌ {result.error[:200]}")
        col.caption(f"⏱ {result.elapsed:.1f}s")
        return

    # Answer
    col.markdown(result.answer)

    # Time badge
    col.caption(f"⏱ **{result.elapsed:.1f}s**")

    # Metadata pills
    if result.metadata:
        pills = []
        if "num_nodes" in result.metadata:
            pills.append(f"🗂 {result.metadata['num_nodes']} nodes")
        if "num_triplets" in result.metadata:
            pills.append(f"🔗 {result.metadata['num_triplets']} triplets")
        if "num_chunks" in result.metadata:
            pills.append(f"📄 {result.metadata['num_chunks']} chunks")
        if pills:
            col.caption("  ·  ".join(pills))

    # GraphRAG retrieval detail (expandable)
    if name == "graphrag" and result.retrieval_detail:
        with col.expander("📊 Chi tiết retrieval", expanded=False):
            _render_retrieval_detail(result.retrieval_detail, key_prefix=key_prefix, compact=True)


# ── Main Render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.title("⚖️ So sánh — 3 hệ thống cùng trả lời")
    st.caption(
        "Nhập một câu hỏi, hệ thống sẽ gửi đồng thời tới GraphRAG, RAG và LLM "
        "rồi hiển thị kết quả cạnh nhau để bạn so sánh."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚖️ So sánh")
    st.sidebar.info(
        "Trang này gửi **cùng 1 câu hỏi** tới cả 3 hệ thống song song:\n\n"
        "- 🔍 **GraphRAG**: Truy xuất Knowledge Graph → LLM\n"
        "- 📚 **RAG**: Tìm văn bản vector → LLM\n"
        "- 💬 **LLM thuần**: Chỉ dùng AI, không data bổ sung"
    )

    if st.sidebar.button("🗑️ Xóa lịch sử"):
        st.session_state.compare_history = []
        from src.utils.history_store import HistoryStore
        HistoryStore.clear(page_filter="compare")
        st.rerun()

    # ── State init ────────────────────────────────────────────────────────────
    if "compare_history" not in st.session_state:
        st.session_state.compare_history = []
    if "compare_prefill" not in st.session_state:
        st.session_state.compare_prefill = ""

    # Restore from persistent storage
    if not st.session_state.compare_history:
        try:
            from src.utils.history_store import HistoryStore
            for e in reversed(HistoryStore.load(page_filter="compare")):
                st.session_state.compare_history.append({
                    "query": e["query"],
                    "results": {
                        "graphrag": _SystemResult(
                            answer=e["metadata"].get("graphrag_answer", ""),
                            elapsed=e["metadata"].get("graphrag_time", 0),
                        ),
                        "rag": _SystemResult(
                            answer=e["metadata"].get("rag_answer", ""),
                            elapsed=e["metadata"].get("rag_time", 0),
                        ),
                        "chat": _SystemResult(
                            answer=e["metadata"].get("chat_answer", ""),
                            elapsed=e["metadata"].get("chat_time", 0),
                        ),
                    },
                })
        except Exception:
            pass

    # ── Onboarding ────────────────────────────────────────────────────────────
    if not st.session_state.compare_history:
        st.info(
            "💡 **Chế độ so sánh** giúp bạn đánh giá nhanh sự khác biệt giữa 3 phương pháp "
            "trả lời: GraphRAG, RAG và LLM thuần.\n\n"
            "Hãy thử hỏi một câu — hoặc chọn câu mẫu bên dưới:"
        )
        cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for col, q in zip(cols, _EXAMPLE_QUESTIONS):
            if col.button(q, use_container_width=True, key=f"cmp_eg_{q[:15]}"):
                st.session_state.compare_prefill = q
                st.rerun()

    # ── History display ───────────────────────────────────────────────────────
    for idx, item in enumerate(st.session_state.compare_history):
        query = item["query"]
        results = item["results"]

        # User question
        with st.chat_message("user"):
            st.markdown(query)

        # 3-column answer
        col_g, col_r, col_c = st.columns(3)
        pfx = f"cmp_h{idx}_"
        _render_column(col_g, "graphrag", results["graphrag"], key_prefix=pfx)
        _render_column(col_r, "rag",      results["rag"],      key_prefix=pfx)
        _render_column(col_c, "chat",     results["chat"],     key_prefix=pfx)

        st.divider()

    # ── Input ─────────────────────────────────────────────────────────────────
    prefill = st.session_state.pop("compare_prefill", "")
    if query := (st.chat_input("Hỏi về Chèo để so sánh 3 hệ thống...") or prefill or "") or None:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)

        # Run all 3 in parallel
        with st.spinner("⏳ Đang chạy song song 3 hệ thống — GraphRAG · RAG · LLM..."):
            results = _run_all(query)

        # Display results in 3 columns
        col_g, col_r, col_c = st.columns(3)
        pfx = f"cmp_new_{len(st.session_state.compare_history)}_"
        _render_column(col_g, "graphrag", results["graphrag"], key_prefix=pfx)
        _render_column(col_r, "rag",      results["rag"],      key_prefix=pfx)
        _render_column(col_c, "chat",     results["chat"],     key_prefix=pfx)

        # Save to session history
        st.session_state.compare_history.append({
            "query": query,
            "results": results,
        })

        # Persist to file
        try:
            from src.utils.history_store import HistoryStore
            HistoryStore.append(
                page="compare",
                query=query,
                answer=f"[GraphRAG] {results['graphrag'].answer[:100]}...",
                metadata={
                    "graphrag_answer": results["graphrag"].answer,
                    "graphrag_time":   round(results["graphrag"].elapsed, 2),
                    "graphrag_error":  results["graphrag"].error,
                    "rag_answer":      results["rag"].answer,
                    "rag_time":        round(results["rag"].elapsed, 2),
                    "rag_error":       results["rag"].error,
                    "chat_answer":     results["chat"].answer,
                    "chat_time":       round(results["chat"].elapsed, 2),
                    "chat_error":      results["chat"].error,
                },
            )
        except Exception:
            pass
