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
    ["💬 Chat", "🔗 Neo4j"],
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
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "💬 Chat":
    render_chat()
elif page == "🔗 Neo4j":
    render_neo4j()
