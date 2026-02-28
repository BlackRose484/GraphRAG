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

from ui import page_chat, page_neo4j, page_graphrag, page_rag, page_benchmark

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
    ["💬 Chat", "🔗 Neo4j", "🔍 GraphRAG", "📚 RAG", "📊 Benchmark"],
    label_visibility="collapsed",
)
st.sidebar.divider()

try:
    from src.utils.logger import current_log_path
    _lp = current_log_path()
    st.sidebar.caption(f"📋 Log: `{_lp.parent.name}/{_lp.name}`")
except Exception:
    pass

# ── Router ────────────────────────────────────────────────────────────────────

if page == "💬 Chat":
    page_chat.render()
elif page == "🔗 Neo4j":
    page_neo4j.render()
elif page == "🔍 GraphRAG":
    page_graphrag.render()
elif page == "📚 RAG":
    page_rag.render()
elif page == "📊 Benchmark":
    page_benchmark.render()
