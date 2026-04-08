"""
GraphRAGv2 — Streamlit entry point.

Chỉ chứa page-config, sidebar nav và router.
Logic của từng trang nằm trong ui/page_*.py.

Chạy: streamlit run main.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from ui import page_home, page_chat, page_neo4j, page_graphrag, page_rag, page_benchmark, page_compare, page_experiment

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Chèo",
    page_icon="🎭",
    layout="wide",
)

# ── Navigation pages ──────────────────────────────────────────────────────────

_PAGES = ["🏠 Giới thiệu", "⚖️ So sánh", "🔍 GraphRAG", "📚 RAG", "💬 Chat", "🔗 Neo4j", "📊 Benchmark", "🧪 Thử nghiệm"]

# Initialise nav state (default: Home)
if "_nav_page" not in st.session_state:
    st.session_state["_nav_page"] = _PAGES[0]

# Keep the radio in sync with programmatic navigation from page_home
_current_idx = _PAGES.index(st.session_state["_nav_page"]) if st.session_state["_nav_page"] in _PAGES else 0

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🎭 GraphRAGv2")
page = st.sidebar.radio(
    "Điều hướng",
    _PAGES,
    index=_current_idx,
    label_visibility="collapsed",
)

# Sync back: if user clicked sidebar radio, update session nav state
st.session_state["_nav_page"] = page

st.sidebar.divider()

try:
    from src.utils.logger import current_log_path
    _lp = current_log_path()
    st.sidebar.caption(f"📋 Log: `{_lp.parent.name}/{_lp.name}`")
except Exception:
    pass

# ── Router ────────────────────────────────────────────────────────────────────

if page == "🏠 Giới thiệu":
    page_home.render()
elif page == "⚖️ So sánh":
    page_compare.render()
elif page == "🔍 GraphRAG":
    page_graphrag.render()
elif page == "📚 RAG":
    page_rag.render()
elif page == "💬 Chat":
    page_chat.render()
elif page == "🔗 Neo4j":
    page_neo4j.render()
elif page == "📊 Benchmark":
    page_benchmark.render()
elif page == "🧪 Thử nghiệm":
    page_experiment.render()
