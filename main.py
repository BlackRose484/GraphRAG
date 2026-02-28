"""
GraphRAGv2 — Streamlit Interface

Pages:
  💬 Chat   — kiểm tra kết nối LLM
  🔗 Neo4j  — kiểm tra kết nối + load ontology vào Neo4j

Chạy: streamlit run main.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="GraphRAGv2",
    page_icon="🎭",
    layout="wide",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🎭 GraphRAGv2")
page = st.sidebar.radio(
    "Điều hướng",
    ["💬 Chat", "🔗 Neo4j", "🔍 GraphRAG"],
    label_visibility="collapsed",
)
st.sidebar.divider()

# Hiển thị log file của lần chạy hiện tại
try:
    from src.utils.logger import current_log_path
    _lp = current_log_path()
    st.sidebar.caption(f"📋 Log: `{_lp.parent.name}/{_lp.name}`")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_llm():
    """Khởi tạo LLM một lần, lưu vào session_state."""
    if "llm_model" not in st.session_state:
        from src.core.settings import settings as _s
        from src.core.base import BaseModel

        class _ChatModel(BaseModel):
            pass

        st.session_state["llm_model"]    = _ChatModel()
        st.session_state["llm_settings"] = _s

    return st.session_state["llm_model"], st.session_state["llm_settings"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════

def render_chat():
    st.title("💬 Chat với AI")
    st.caption("Powered by LiteLLM — provider-agnostic")

    llm_ready = False
    try:
        chat_model, cfg = _load_llm()
        api_ok, api_msg = cfg.llm.validate_api_key()

        st.sidebar.success("✅ LLM kết nối OK")
        st.sidebar.markdown(f"**Provider:** `{cfg.llm.provider}`")
        st.sidebar.markdown(f"**Model:** `{cfg.llm.model}`")
        st.sidebar.markdown(f"**Temperature:** `{cfg.llm.temperature}`")
        if not api_ok:
            st.sidebar.warning(f"⚠️ {api_msg}")
        llm_ready = True

    except Exception as exc:
        st.error(f"❌ Không thể khởi tạo LLM: {exc}")
        st.info("Kiểm tra file `.env` — đảm bảo API key đã được cấu hình.")
        return

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Cài đặt")
    system_prompt = st.sidebar.text_area(
        "System prompt",
        value=(
            "Bạn là trợ lý chuyên về nghệ thuật Chèo Việt Nam. "
            "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác."
        ),
        height=120,
    )
    if st.sidebar.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Hỏi về Chèo...", disabled=not llm_ready):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                try:
                    history_text = ""
                    for m in st.session_state.messages[:-1]:
                        role_label = "Người dùng" if m["role"] == "user" else "Trợ lý"
                        history_text += f"{role_label}: {m['content']}\n"

                    full_prompt = (
                        f"{system_prompt}\n\n"
                        f"{history_text}"
                        f"Người dùng: {prompt}\n"
                        f"Trợ lý:"
                    )
                    answer = chat_model.safe_generate(full_prompt)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as exc:
                    err = f"❌ Lỗi khi gọi LLM: {exc}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEO4J
# ══════════════════════════════════════════════════════════════════════════════

def render_neo4j():
    st.title("🔗 Neo4j — Kiểm tra kết nối")

    from src.core.settings import settings

    with st.expander("⚙️ Cấu hình kết nối", expanded=True):
        col1, col2 = st.columns(2)
        col1.markdown(f"**URI:** `{settings.neo4j.uri}`")
        col1.markdown(f"**User:** `{settings.neo4j.user}`")
        col2.markdown(
            f"**Password:** {'✅ Đã cấu hình' if settings.neo4j.password else '❌ Chưa cấu hình'}"
        )
        col2.markdown(f"**Ontology file:** `{settings.ontology.file_path.name}`")

    st.divider()

    if st.button("🔍 Kiểm tra kết nối", use_container_width=False):
        from src.graph_loader.neo4j_client import Neo4jClient
        client = Neo4jClient()
        try:
            with st.spinner("Đang ping Neo4j..."):
                client.ping()
            st.success("✅ Kết nối Neo4j thành công!")

            with st.spinner("Đang lấy thông tin graph..."):
                schema = client.get_schema_summary()

            m1, m2, m3 = st.columns(3)
            m1.metric("Tổng nodes", schema["total_nodes"])
            m2.metric("Tổng relationships", schema["total_relationships"])
            m3.metric("Node labels", len(schema["node_labels"]))

            if schema["node_labels"]:
                st.markdown("**Labels:** " + ", ".join(
                    f"`{lb}`" for lb in sorted(schema["node_labels"])
                ))
            if schema["relationship_types"]:
                st.markdown("**Relationship types:** " + ", ".join(
                    f"`{r}`" for r in sorted(schema["relationship_types"])
                ))
        except Exception as exc:
            st.error(f"❌ {exc}")
        finally:
            client.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GRAPHRAG
# ══════════════════════════════════════════════════════════════════════════════

def _get_pipeline(
    strategy: str,
    retrieval_methods: list,
    format_keys: list,
    enable_enhancement: bool,
):
    """Create (or reuse) a GraphRAGPipeline stored in session_state."""
    cache_key = f"pipeline_{strategy}_{'_'.join(retrieval_methods)}"
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


def render_graphrag():
    st.title("🔍 GraphRAG — Hỏi đáp về Chèo")
    st.caption("Graph-Retrieval Augmented Generation: kết hợp knowledge graph + LLM")

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Cài đặt pipeline")

    from src.constants.constant import (
        FormatKey, GenerationStrategy, RetrievalMethod
    )

    strategy = st.sidebar.selectbox(
        "Generation strategy",
        options=[GenerationStrategy.PRE, GenerationStrategy.MID, GenerationStrategy.POST],
        index=1,
        format_func=lambda s: {"pre": "Pre (context first)",
                               "mid": "Mid (structured guidance)",
                               "post": "Post (verify & refine)"}[s],
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
        "Query enhancement", value=True,
        help="Expand / decompose complex queries before retrieval"
    )

    if st.sidebar.button("🗑️ Xóa lịch sử"):
        st.session_state.rag_history = []
        st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    for item in st.session_state.rag_history:
        with st.chat_message("user"):
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander("📊 Chi tiết retrieval"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Nodes",      item["num_nodes"])
                col2.metric("Triplets",   item["num_triplets"])
                col3.metric("Paths",      item["num_paths"])
                st.caption(
                    f"Retrieval: {item['retrieval_time']:.2f}s | "
                    f"Generation: {item['gen_time']:.2f}s | "
                    f"Total: {item['total_time']:.2f}s"
                )

    # ── Input ─────────────────────────────────────────────────────────────────
    if not retrieval_methods:
        st.warning("Chọn ít nhất một retrieval method ở sidebar.")
        return
    if not format_keys:
        st.warning("Chọn ít nhất một context format ở sidebar.")
        return

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

                    st.session_state.rag_history.append({
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


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "💬 Chat":
    render_chat()
elif page == "🔗 Neo4j":
    render_neo4j()
elif page == "🔍 GraphRAG":
    render_graphrag()
