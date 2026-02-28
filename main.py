"""
GraphRAGv2 — Streamlit Chat Interface

Giao diện chat đơn giản để kiểm tra kết nối LLM qua LiteLLM.
Chạy: streamlit run main.py
"""

from __future__ import annotations

import sys
import os

# Đảm bảo project root trong sys.path khi chạy từ bất kỳ đâu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAGv2 Chat",
    page_icon="🎭",
    layout="centered",
)

# ── Load settings & base model (session-state cached) ────────────────────────

def load_llm():
    """Khởi tạo một lần duy nhất, lưu vào session_state."""
    if "llm_model" not in st.session_state:
        from src.core.settings import settings as _settings
        from src.core.base import BaseModel

        class _ChatModel(BaseModel):
            """Minimal subclass — chỉ dùng safe_generate."""
            pass

        st.session_state["llm_model"] = _ChatModel()
        st.session_state["llm_settings"] = _settings

    return st.session_state["llm_model"], st.session_state["llm_settings"]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎭 GraphRAGv2 Chat")
st.caption("Giao diện kiểm tra kết nối LLM — Chèo Knowledge Graph")

# ── Load model ────────────────────────────────────────────────────────────────

try:
    chat_model, settings = load_llm()
    provider = settings.llm.provider
    model_name = settings.llm.model

    st.sidebar.success("✅ LLM đã kết nối")
    st.sidebar.markdown(f"**Provider:** `{provider}`")
    st.sidebar.markdown(f"**Model:** `{model_name}`")
    st.sidebar.markdown(f"**Temperature:** `{settings.llm.temperature}`")
    st.sidebar.markdown(f"**Timeout:** `{settings.llm.timeout}s`")
    st.sidebar.divider()

    api_ok, msg = settings.llm.validate_api_key()
    if not api_ok:
        st.sidebar.warning(f"⚠️ {msg}")

    llm_ready = True

except Exception as e:
    st.error(f"❌ Không thể khởi tạo LLM: {e}")
    st.info("Kiểm tra file `.env` và đảm bảo API key đã được cấu hình.")
    llm_ready = False

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.markdown("### ⚙️ Cài đặt chat")

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

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input(
    "Hỏi về Chèo...",
    disabled=not llm_ready,
):
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi LLM
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Gộp system prompt + lịch sử + câu hỏi mới thành 1 prompt
                history_text = ""
                for m in st.session_state.messages[:-1]:  # bỏ câu hỏi vừa add
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

            except Exception as e:
                err_msg = f"❌ Lỗi khi gọi LLM: {e}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )
