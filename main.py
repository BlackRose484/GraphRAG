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
import streamlit.components.v1 as _components

from ui import page_home, page_chat, page_neo4j, page_graphrag, page_rag, page_benchmark, page_compare, page_experiment, page_preference
from ui.model_selector import render_global_model_selector

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Chèo",
    page_icon="🎭",
    layout="wide",
)

# ── Tab persistence (inject once per browser session) ─────────────────────────
# Fixes "need 2-3 clicks to switch tab" by saving active tab index to
# sessionStorage and restoring it after every Streamlit rerender.

_components.html("""
<script>
(function () {
    var p = window.parent;
    if (p.__stab_init) return;
    p.__stab_init = true;

    var doc = p.document;
    var ss  = p.sessionStorage;
    var userClickedAt = 0;

    function tabKey(tl) {
        var labels = Array.from(tl.querySelectorAll('[role="tab"]'))
            .map(function(t) { return t.textContent.trim().slice(0, 20); }).join('|');
        return 'stab__' + labels.slice(0, 80);
    }

    doc.addEventListener('click', function (e) {
        if (!e.isTrusted) return;
        var tab = e.target.closest('[role="tab"]');
        if (!tab) return;
        var tl = tab.closest('[role="tablist"]');
        if (!tl) return;
        var idx = Array.from(tl.querySelectorAll('[role="tab"]')).indexOf(tab);
        if (idx >= 0) {
            ss.setItem(tabKey(tl), String(idx));
            userClickedAt = Date.now();
        }
    }, true);

    var timer;
    new MutationObserver(function () {
        clearTimeout(timer);
        timer = setTimeout(function () {
            if (Date.now() - userClickedAt < 400) return;
            doc.querySelectorAll('[role="tablist"]').forEach(function (tl) {
                var raw = ss.getItem(tabKey(tl));
                if (raw === null) return;
                var saved = parseInt(raw, 10);
                if (isNaN(saved) || saved < 0) return;
                var tabs = Array.from(tl.querySelectorAll('[role="tab"]'));
                if (saved >= tabs.length) return;
                var curr = tabs.findIndex(function(t) {
                    return t.getAttribute('aria-selected') === 'true';
                });
                if (curr !== saved) tabs[saved].click();
            });
        }, 150);
    }).observe(doc.body, { childList: true, subtree: true });

    doc.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter') return;
        var active = doc.activeElement;
        if (!active) return;
        var form = active.closest('[data-testid="stForm"]');
        if (!form) return;
        if (active.tagName === 'TEXTAREA') return;
        var btn = form.querySelector('button[kind="primaryFormSubmit"], button[data-testid="baseButton-primaryFormSubmit"]');
        if (btn) { e.preventDefault(); btn.click(); }
    }, true);
})();
</script>
""", height=0)

# ── Navigation pages ──────────────────────────────────────────────────────────

_PAGES = ["🏠 Giới thiệu", "⚖️ So sánh", "🔍 GraphRAG", "📚 RAG", "💬 Chat", "🔗 Neo4j", "📊 Benchmark", "🧪 Thử nghiệm", "📋 Đánh giá ưu tiên"]

# Initialise nav state (default: Home)
if "_nav_radio" not in st.session_state:
    st.session_state["_nav_radio"] = _PAGES[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🎭 GraphRAGv2")
page = st.sidebar.radio(
    "Điều hướng",
    _PAGES,
    key="_nav_radio",
    label_visibility="collapsed",
)

# Sync back
st.session_state["_nav_page"] = page

st.sidebar.divider()

# Global LLM model selector — applies to every page in this rerun.
render_global_model_selector()

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
elif page == "📋 Đánh giá ưu tiên":
    page_preference.render()
