"""
🕐 History page — Lịch sử hội thoại tổng hợp.

Hiển thị toàn bộ lịch sử câu hỏi/trả lời từ 3 trang (Chat, RAG, GraphRAG)
theo thứ tự thời gian ngược (mới nhất trước).
"""
from __future__ import annotations

import streamlit as st

# Page badges
_PAGE_META = {
    "graphrag": {"label": "🔍 GraphRAG", "color": "#c05000", "bg": "#fff0e0"},
    "rag":      {"label": "📚 RAG",      "color": "#1565c0", "bg": "#e8f0fe"},
    "chat":     {"label": "💬 Chat",     "color": "#1b6b3a", "bg": "#e8f5e9"},
}


def _badge(page: str) -> str:
    m = _PAGE_META.get(page, {"label": page, "color": "#666", "bg": "#eee"})
    return (
        f"<span style='"
        f"background:{m['bg']};color:{m['color']};"
        f"border:1px solid {m['color']}44;border-radius:4px;"
        f"padding:2px 8px;font-size:0.75rem;font-weight:600"
        f"'>{m['label']}</span>"
    )


def _fmt_ts(ts: str) -> str:
    """Format ISO timestamp to readable Vietnamese format."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return ts


def render() -> None:
    from src.utils.history_store import HistoryStore

    st.title("🕐 Lịch sử hội thoại")
    st.caption("Toàn bộ câu hỏi đã đặt trong các phiên — được lưu lại tự động.")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔎 Bộ lọc")

    page_options = {
        "Tất cả": None,
        "🔍 GraphRAG": "graphrag",
        "📚 RAG": "rag",
        "💬 Chat": "chat",
    }
    selected_label = st.sidebar.radio(
        "Trang nguồn",
        list(page_options.keys()),
        index=0,
        label_visibility="collapsed",
    )
    page_filter = page_options[selected_label]

    search_kw = st.sidebar.text_input(
        "🔍 Tìm kiếm câu hỏi",
        placeholder="Nhập từ khoá...",
    )

    st.sidebar.divider()

    # Count stats
    counts = HistoryStore.count()
    total = counts.get("total", 0)
    st.sidebar.markdown("### 📊 Thống kê")
    col_s1, col_s2 = st.sidebar.columns(2)
    col_s1.metric("Tổng", total)
    col_s2.metric("GraphRAG", counts.get("graphrag", 0))
    col_s3, col_s4 = st.sidebar.columns(2)
    col_s3.metric("RAG", counts.get("rag", 0))
    col_s4.metric("Chat", counts.get("chat", 0))

    st.sidebar.divider()

    # Clear buttons
    st.sidebar.markdown("### 🗑️ Xóa lịch sử")
    if st.sidebar.button("Xóa tất cả", use_container_width=True):
        n = HistoryStore.clear()
        st.sidebar.success(f"Đã xóa {n} mục")
        st.rerun()

    if page_filter:
        label_txt = selected_label.split(" ", 1)[-1]
        if st.sidebar.button(f"Xóa {label_txt}", use_container_width=True):
            n = HistoryStore.clear(page_filter=page_filter)
            st.sidebar.success(f"Đã xóa {n} mục")
            st.rerun()

    # ── Load entries ──────────────────────────────────────────────────────────
    entries = HistoryStore.load(page_filter=page_filter)

    if search_kw:
        kw_lower = search_kw.lower()
        entries = [
            e for e in entries
            if kw_lower in e.get("query", "").lower()
            or kw_lower in e.get("answer", "").lower()
        ]

    # ── Empty state ───────────────────────────────────────────────────────────
    if not entries:
        st.info(
            "Chưa có lịch sử nào.\n\n"
            "Hãy đặt câu hỏi ở trang **🔍 GraphRAG**, **📚 RAG** hoặc **💬 Chat** — "
            "lịch sử sẽ tự động được lưu lại tại đây."
        )
        return

    st.markdown(f"**{len(entries)}** mục{'  ·  _đang lọc_' if (page_filter or search_kw) else ''}")
    st.divider()

    # ── Entry list ────────────────────────────────────────────────────────────
    for entry in entries:
        page  = entry.get("page", "")
        query = entry.get("query", "")
        answer = entry.get("answer", "")
        ts    = _fmt_ts(entry.get("timestamp", ""))
        meta  = entry.get("metadata", {})

        # Header row
        col_ts, col_badge, col_btn = st.columns([3, 2, 1])
        col_ts.markdown(
            f"<span style='color:#6b7280;font-size:0.85rem'>🕐 {ts}</span>",
            unsafe_allow_html=True,
        )
        col_badge.markdown(_badge(page), unsafe_allow_html=True)

        # Quick-jump button: send to the right page with prefill
        _PREFILL_KEY = {
            "graphrag": "graphrag_prefill",
            "rag":      "rag_prefill",
            "chat":     "chat_prefill",
        }
        _PAGE_NAV = {
            "graphrag": "🔍 GraphRAG",
            "rag":      "📚 RAG",
            "chat":     "💬 Chat",
        }
        if col_btn.button("↩ Hỏi lại", key=f"re_{entry.get('id', ts)}", use_container_width=True):
            prefill_key = _PREFILL_KEY.get(page, "")
            nav_page    = _PAGE_NAV.get(page, "🔍 GraphRAG")
            if prefill_key:
                st.session_state[prefill_key] = query
            st.session_state["_nav_page"] = nav_page
            st.rerun()

        # Question
        st.markdown(
            f"<div style='font-weight:600;font-size:0.97rem;margin:4px 0 6px'>{query}</div>",
            unsafe_allow_html=True,
        )

        # Answer (truncated, expandable)
        short = answer[:300].rstrip()
        is_long = len(answer) > 300
        with st.expander(
            f"{'📄 ' + short[:80] + '…' if is_long else '📄 ' + short[:80]}",
            expanded=False,
        ):
            st.markdown(answer)

            # Metadata pills
            if meta:
                pills = []
                if "num_nodes" in meta:
                    pills.append(f"🗂 {meta['num_nodes']} nodes")
                if "num_triplets" in meta:
                    pills.append(f"🔗 {meta['num_triplets']} triplets")
                if "retrieval_time" in meta:
                    pills.append(f"⏱ retrieval {meta['retrieval_time']:.1f}s")
                if "total_time" in meta:
                    pills.append(f"⏱ total {meta['total_time']:.1f}s")
                if "strategy" in meta:
                    pills.append(f"⚙️ {meta['strategy']}")
                if pills:
                    st.caption("  ·  ".join(pills))

        st.markdown(
            "<hr style='margin:6px 0;border:none;border-top:1px solid #f0f0f0'>",
            unsafe_allow_html=True,
        )
