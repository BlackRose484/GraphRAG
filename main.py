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

from ui import page_home, page_chat, page_neo4j, page_graphrag, page_rag, page_benchmark, page_compare, page_experiment, page_preference, page_intro
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

_ALL_PAGES   = ["🏠 Giới thiệu", "👋 Chào mừng", "⚖️ So sánh", "🔍 GraphRAG", "📚 RAG", "💬 Chat", "🔗 Neo4j", "📊 Benchmark", "🧪 Thử nghiệm", "📋 Đánh giá ưu tiên"]
_GUEST_PAGES = ["👋 Chào mừng", "🧪 Thử nghiệm", "📋 Đánh giá ưu tiên"]

# ── Guest mode + admin unlock ────────────────────────────────────────────────
# Behaviour:
#   GUEST_MODE unset     → full app always
#   GUEST_MODE=1         → only _GUEST_PAGES until admin password entered
#   ADMIN_PASSWORD unset → no unlock path (public-only deployment)

from src.core.settings import settings as _app_settings  # noqa: E402


def _is_admin() -> bool:
    """Is the current Streamlit session allowed to see admin-only pages?"""
    if not _app_settings.app.guest_mode:
        return True
    return bool(st.session_state.get("_admin_unlocked"))


def _render_admin_unlock() -> None:
    """Password prompt to unlock admin pages, shown only in guest mode."""
    if not _app_settings.app.guest_mode:
        return
    if st.session_state.get("_admin_unlocked"):
        if st.sidebar.button("🔒 Khóa lại", use_container_width=True):
            st.session_state["_admin_unlocked"] = False
            st.session_state.pop("_nav_radio", None)  # reset nav
            st.rerun()
        return
    if not _app_settings.app.admin_unlock_available:
        return  # No password configured — silent
    with st.sidebar.expander("🔑 Admin", expanded=False):
        pw = st.text_input("Mật khẩu", type="password", key="_admin_pw_input")
        if st.button("Mở khóa", key="_admin_unlock_btn"):
            if pw and pw == _app_settings.app.admin_password:
                st.session_state["_admin_unlocked"] = True
                st.session_state.pop("_nav_radio", None)  # reset nav
                st.rerun()
            else:
                st.error("Sai mật khẩu")


available_pages = _ALL_PAGES if _is_admin() else _GUEST_PAGES

# Apply pending navigation from page buttons (must run BEFORE the radio widget
# is instantiated — Streamlit blocks writes to a widget-backed key after that).
_pending = st.session_state.pop("_nav_pending", None)
if _pending and _pending in available_pages:
    st.session_state["_nav_radio"] = _pending

# If session state points to a page that's no longer visible, snap to first.
if st.session_state.get("_nav_radio") not in available_pages:
    st.session_state["_nav_radio"] = available_pages[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🎭 GraphRAGv2")
page = st.sidebar.radio(
    "Điều hướng",
    available_pages,
    key="_nav_radio",
    label_visibility="collapsed",
)
st.session_state["_nav_page"] = page

st.sidebar.divider()

# Admin controls appear in guest mode (unlock) or admin mode (lock).
_render_admin_unlock()

# Everything below is admin-only UI — keep the guest sidebar minimal.
if _is_admin():
    st.sidebar.divider()
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
elif page == "👋 Chào mừng":
    page_intro.render()
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
