"""
Global LLM model selector — one sidebar dropdown, shared by every page.

On every Streamlit rerun, ``render_global_model_selector`` reads the allowlist
from ``settings.llm.available_models`` (auto-detect by API keys, see
:mod:`src.core.settings`), shows a dropdown, and mutates
``settings.llm.model`` to the chosen value. Because
:class:`src.core.base.BaseModel` resolves ``model_name`` dynamically from
``settings.llm.model``, every pipeline / metric / chat call downstream in the
same rerun picks up the choice automatically — no per-page plumbing needed.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

_SESSION_KEY = "global_llm_model"


def render_global_model_selector() -> str:
    """Render the selector in the current sidebar and return the chosen model.

    Safe to call from ``main.py`` before page routing: sets
    ``settings.llm.model`` for the rest of this rerun.
    """
    from src.core.settings import settings

    models = settings.llm.available_models
    current = settings.llm.model

    st.sidebar.markdown("### 🧠 LLM model")

    if not models:
        st.sidebar.warning("Không có model nào configured. Kiểm tra `.env`.")
        return current

    # Restore prior choice from session, else fall back to current settings.
    saved: Optional[str] = st.session_state.get(_SESSION_KEY)
    default = saved if saved in models else current
    try:
        default_idx = models.index(default)
    except ValueError:
        default_idx = 0

    source = settings.llm.available_models_source
    source_label = "auto từ API keys" if source == "auto" else "từ LLM_MODELS_AVAILABLE"

    chosen = st.sidebar.selectbox(
        f"Model ({source_label})",
        models,
        index=default_idx,
        key="_global_model_selectbox",
        help=(
            "Model áp dụng cho toàn bộ ứng dụng: Chat, GraphRAG, RAG, "
            "Benchmark, Experiment. Thay đổi ở đây → ngay lập tức mọi "
            "pipeline đang chạy trong session sẽ dùng model mới. "
            "`.env` KHÔNG bị sửa."
        ),
    )
    st.session_state[_SESSION_KEY] = chosen

    # Mutate settings so every downstream BaseModel resolves to the new model.
    if settings.llm.model != chosen:
        settings.llm.model = chosen

    return chosen
