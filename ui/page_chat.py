"""
💬 Chat page — kiểm tra / dùng thử LLM trực tiếp.
"""

from __future__ import annotations

import streamlit as st


# ── Helper ────────────────────────────────────────────────────────────────────

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


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
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
