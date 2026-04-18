"""
📋 Preference page — đánh giá ưu tiên: chọn câu trả lời tốt nhất.

Luồng: Nhập tên → Đọc 21 câu hỏi (pre-generated) → Chọn hệ thống tốt nhất
     → Lưu & Next → Xem tổng kết khi hoàn thành.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.core.settings import settings

# ── Paths ─────────────────────────────────────────────────────────────────────

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "results"
_DATASETS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "datasets"
_ANSWERS_PATH = _DATASETS_DIR / "pregenerated_answers.json"

# ── Display config ────────────────────────────────────────────────────────────

_DISPLAY_ORDER = ["chat", "rag", "graphrag"]
_DISPLAY_NAMES = {
    "chat": "Hệ thống AI",
    "rag": "Hệ thống AI Plus",
    "graphrag": "Hệ thống AI Pro",
}
_DISPLAY_ICONS = {
    "chat": "🤖",
    "rag": "📚",
    "graphrag": "🕸️",
}
_DISPLAY_COLORS = {
    "chat": "#e67e22",
    "rag": "#3498db",
    "graphrag": "#2ecc71",
}
_DISPLAY_DESCS = {
    "chat": "Mô hình ngôn ngữ<br>trả lời trực tiếp",
    "rag": "Tìm kiếm ngữ nghĩa<br>+ Tổng hợp văn bản",
    "graphrag": "Knowledge Graph<br>+ Mô hình ngôn ngữ",
}

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Hero ── */
.pref-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    border-radius: 16px; padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem; text-align: center;
}
.pref-hero-title {
    font-size: 1.9rem; font-weight: 800; color: #ffffff;
    margin: 0 0 0.4rem; letter-spacing: -0.3px;
}
.pref-hero-sub {
    font-size: 1rem; color: rgba(255,255,255,0.75); margin: 0;
}

/* ── System cards ── */
.pref-sys-cards { display: flex; gap: 12px; margin: 1.4rem 0 0.5rem; }
.pref-sys-card {
    flex: 1; border-radius: 12px; padding: 1.1rem 0.8rem;
    text-align: center; color: white;
}
.pref-sys-card-icon  { font-size: 2rem; margin-bottom: 6px; }
.pref-sys-card-name  { font-weight: 700; font-size: 0.95rem; }
.pref-sys-card-desc  { font-size: 0.75rem; opacity: 0.85; margin-top: 4px; line-height: 1.3; }

/* ── Hint pills ── */
.pref-hint-row {
    display: flex; gap: 18px; justify-content: center;
    margin-top: 1rem; flex-wrap: wrap;
}
.pref-hint-pill {
    background: rgba(255,255,255,0.12); border-radius: 20px;
    padding: 4px 14px; font-size: 0.8rem; color: rgba(255,255,255,0.88);
}

/* ── Question card ── */
.q-header {
    background: linear-gradient(90deg, #f0f4ff 0%, #fafbff 100%);
    border: 1px solid #dde6f5; border-left: 4px solid #2980b9;
    border-radius: 0 10px 10px 0; padding: 0.8rem 1.2rem;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
}
.q-header-num  { font-size: 1.5rem; font-weight: 800; color: #2c3e7f; }
.q-header-body { flex: 1; }
.q-header-title{ font-size: 1rem; font-weight: 700; color: #1a2a5e; margin: 0; }
.q-header-sub  { font-size: 0.78rem; color: #888; margin: 2px 0 0; }
.q-cat-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; color: white;
}

/* ── Column header ── */
.pref-col-hdr {
    border-radius: 8px; padding: 9px 14px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 8px;
}
.pref-col-hdr-title { font-weight: 700; font-size: 1rem; color: white; }

/* ── Answer container ── */
.answer-box {
    background: #fafbfc; border: 1px solid #e5e9f0;
    border-radius: 8px; padding: 12px 14px;
    max-height: 450px; overflow-y: auto;
    font-size: 0.88rem; line-height: 1.6;
}

/* ── Rating section ── */
.pref-rating-wrap {
    background: #f8faff; border: 1px solid #e0e8f5;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-top: 1rem;
}
.pref-rating-title {
    font-size: 1rem; font-weight: 700; color: #1a2a5e;
    margin-bottom: 1rem; border-bottom: 2px solid #e0e8f5; padding-bottom: 0.5rem;
}

/* ── Completion banner ── */
.pref-complete-banner {
    background: linear-gradient(135deg, #0d7a45 0%, #27ae60 60%, #2ecc71 100%);
    border-radius: 16px; padding: 2rem 2rem 1.6rem;
    text-align: center; margin-bottom: 1.4rem;
}
.pref-complete-title { font-size: 1.8rem; font-weight: 800; color: white; margin: 0 0 0.3rem; }
.pref-complete-sub   { font-size: 1rem; color: rgba(255,255,255,0.88); margin: 0; }
.pref-complete-meta  { margin-top: 1rem; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
.pref-complete-chip  {
    background: rgba(255,255,255,0.18); border-radius: 20px;
    padding: 4px 16px; font-size: 0.82rem; color: white;
}

/* ── Score summary ── */
.pref-score-grid { display: flex; gap: 10px; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
.pref-score-card {
    flex: 1; min-width: 110px; background: white;
    border: 1px solid #e5e9f0; border-radius: 10px;
    padding: 0.8rem 0.6rem; text-align: center;
}
.pref-score-sys   { font-size: 0.72rem; color: #888; font-weight: 600; text-transform: uppercase; }
.pref-score-val   { font-size: 1.8rem; font-weight: 800; color: #1a2a5e; line-height: 1.1; }
.pref-score-label { font-size: 0.68rem; color: #aaa; }

/* ── Sidebar question list ── */
.pref-sb-q {
    font-size: 0.82rem; padding: 2px 0; line-height: 1.4;
    border-bottom: 1px solid #f0f0f0;
}
</style>
"""

# ── Category colors ───────────────────────────────────────────────────────────

_CAT_COLORS: dict[str, tuple[str, str]] = {
    "Dạng 1": ("#27ae60", "Dễ"),
    "Dạng 2": ("#f39c12", "Trung bình"),
    "Dạng 3": ("#e74c3c", "Khó"),
}


def _cat_badge(category: str) -> tuple[str, str]:
    for prefix, (color, label) in _CAT_COLORS.items():
        if category.startswith(prefix):
            return color, label
    return "#888", "?"


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _q_header_html(idx: int, total: int, question: str, category: str) -> str:
    color, label = _cat_badge(category)
    return (
        f'<div class="q-header" style="border-left-color:{color}">'
        f'  <div class="q-header-num">{idx + 1}</div>'
        f'  <div class="q-header-body">'
        f'    <div class="q-header-title">{question}</div>'
        f'    <div class="q-header-sub">Câu {idx + 1} / {total}</div>'
        f'  </div>'
        f'  <span class="q-cat-badge" style="background:{color}">{label}</span>'
        f'</div>'
    )


def _col_header_html(sys_key: str) -> str:
    color = _DISPLAY_COLORS[sys_key]
    icon = _DISPLAY_ICONS[sys_key]
    name = _DISPLAY_NAMES[sys_key]
    return (
        f'<div class="pref-col-hdr" style="background:{color}">'
        f'  <span style="font-size:1.2rem">{icon}</span>'
        f'  <span class="pref-col-hdr-title">{name}</span>'
        f'</div>'
    )


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def _load_data() -> list[dict[str, Any]]:
    if not _ANSWERS_PATH.exists():
        return []
    raw = json.loads(_ANSWERS_PATH.read_text(encoding="utf-8"))
    return raw.get("questions", [])


# ── State management ──────────────────────────────────────────────────────────

def _init_state(total: int) -> None:
    if "pref_user" not in st.session_state:
        st.session_state.pref_user = ""
    if "pref_current" not in st.session_state:
        st.session_state.pref_current = 0
    if "pref_answers" not in st.session_state:
        st.session_state.pref_answers = [
            {"best_system": "", "note": "", "done": False}
            for _ in range(total)
        ]
    if "pref_submitted" not in st.session_state:
        st.session_state.pref_submitted = False


def _done_count() -> int:
    return sum(1 for a in st.session_state.pref_answers if a["done"])


def _first_undone() -> int:
    for i, a in enumerate(st.session_state.pref_answers):
        if not a["done"]:
            return i
    return len(st.session_state.pref_answers)


# ── Save results ──────────────────────────────────────────────────────────────

def _save_file(questions: list[dict[str, Any]]) -> Path:
    user = st.session_state.pref_user or "anonymous"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / f"preference_{user}.json"

    answers_out: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        a = st.session_state.pref_answers[i]
        answers_out.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "best_system": a["best_system"],
            "best_display_name": _DISPLAY_NAMES.get(a["best_system"], ""),
            "note": a["note"],
            "done": a["done"],
        })

    data = {
        "user": user,
        "last_updated": datetime.now().isoformat(),
        "total_questions": len(questions),
        "completed": _done_count(),
        "answers": answers_out,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    questions = _load_data()
    if not questions:
        st.error(
            "Chưa có dữ liệu câu trả lời pre-generated. "
            "Hãy chạy `python -m benchmark.generate_answers` trước."
        )
        return

    total = len(questions)
    _init_state(total)

    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    done = _done_count()
    st.sidebar.markdown("### 📋 Tiến độ đánh giá")
    st.sidebar.progress(done / total, text=f"{done} / {total} câu hoàn thành")

    if st.session_state.pref_user:
        st.sidebar.markdown("")
        for i, q in enumerate(questions):
            icon = "✅" if st.session_state.pref_answers[i]["done"] else "🔲"
            color, label = _cat_badge(q["category"])
            short_q = q["question"][:40] + ("..." if len(q["question"]) > 40 else "")

            if st.sidebar.button(
                f"{icon} {i+1}. {short_q}",
                key=f"pref_nav_{i}",
                use_container_width=True,
            ):
                st.session_state.pref_current = i
                st.rerun()

    if done > 0 and st.session_state.pref_user:
        st.sidebar.markdown("---")
        path = _save_file(questions)
        with open(path, "r", encoding="utf-8") as f:
            st.sidebar.download_button(
                "Tải kết quả JSON", f.read(),
                file_name=path.name, mime="application/json",
            )
    if st.session_state.pref_user:
        if st.sidebar.button("Làm lại từ đầu"):
            for k in [k for k in st.session_state if k.startswith("pref_")]:
                del st.session_state[k]
            st.rerun()

    # ── Gate: nhập tên ───────────────────────────────────────────────────────
    if not st.session_state.pref_user:
        st.markdown(
            '<div class="pref-hero">'
            '  <div class="pref-hero-title">📋 Đánh giá câu trả lời</div>'
            '  <div class="pref-hero-sub">Chọn câu trả lời tốt nhất trong 3 hệ thống AI về Nghệ thuật Chèo</div>'
            '  <div class="pref-sys-cards">'
            + "".join(
                f'<div class="pref-sys-card" style="background:rgba({_hex_to_rgb(_DISPLAY_COLORS[sk])},0.85)">'
                f'  <div class="pref-sys-card-icon">{_DISPLAY_ICONS[sk]}</div>'
                f'  <div class="pref-sys-card-name">{_DISPLAY_NAMES[sk]}</div>'
                f'  <div class="pref-sys-card-desc">{_DISPLAY_DESCS[sk]}</div>'
                f'</div>'
                for sk in _DISPLAY_ORDER
            )
            + '  </div>'
            '  <div class="pref-hint-row">'
            f'    <span class="pref-hint-pill">📋 {total} câu hỏi</span>'
            '    <span class="pref-hint-pill">15-25 phút</span>'
            '    <span class="pref-hint-pill">Tự động lưu</span>'
            '    <span class="pref-hint-pill">Ẩn danh</span>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            st.markdown(
                "<div style='text-align:center;font-weight:700;font-size:1.05rem;"
                "color:#1a2a5e;margin-bottom:8px'>Nhập tên để bắt đầu</div>",
                unsafe_allow_html=True,
            )
            with st.form("pref_gate_form"):
                name = st.text_input(
                    "Tên", placeholder="Ví dụ: Nguyễn Văn A",
                    key="pref_gate_name", label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "Bắt đầu đánh giá",
                    type="primary", use_container_width=True,
                )
            if submitted:
                if name.strip():
                    st.session_state.pref_user = name.strip()
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập tên trước khi bắt đầu.")
        return

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:0.9rem;color:#888;margin-bottom:4px'>"
        f"Xin chào, <strong>{st.session_state.pref_user}</strong></div>",
        unsafe_allow_html=True,
    )

    current = st.session_state.pref_current

    # ── All done → completion screen ─────────────────────────────────────────
    if done >= total:
        _render_completion(questions)
        return

    # ── Current question ─────────────────────────────────────────────────────
    q = questions[current]
    a = st.session_state.pref_answers[current]

    st.markdown(
        _q_header_html(current, total, q["question"], q["category"]),
        unsafe_allow_html=True,
    )

    # Show 3 answers in columns
    cols = st.columns(3)
    for col, sk in zip(cols, _DISPLAY_ORDER):
        with col:
            col.markdown(_col_header_html(sk), unsafe_allow_html=True)
            ans_data = q.get("answers", {}).get(sk, {})
            answer_text = ans_data.get("answer", "") if isinstance(ans_data, dict) else ""
            if answer_text:
                col.markdown(
                    f'<div class="answer-box">{_md_safe(answer_text)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                col.warning("Không có câu trả lời")

    # Rating section
    st.markdown(
        '<div class="pref-rating-wrap">'
        '<div class="pref-rating-title">Chọn câu trả lời tốt nhất</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    choice_labels = [_DISPLAY_NAMES[sk] for sk in _DISPLAY_ORDER]
    default_idx = 0
    if a["done"] and a["best_system"]:
        try:
            default_idx = _DISPLAY_ORDER.index(a["best_system"])
        except ValueError:
            default_idx = 0

    best_label = st.radio(
        "Bạn thích câu trả lời nào nhất?",
        choice_labels,
        index=default_idx,
        key=f"pref_choice_{current}",
        horizontal=True,
    )
    note = st.text_area(
        "Ghi chú (tùy chọn)",
        value=a["note"] if a["done"] else "",
        key=f"pref_note_{current}",
        placeholder="Nhận xét thêm về chất lượng câu trả lời...",
        height=80,
    )

    # Navigation buttons
    c_prev, c_save, c_next = st.columns([1, 2, 1])
    with c_prev:
        if current > 0:
            if st.button("Câu trước", key=f"pref_prev_{current}", use_container_width=True):
                st.session_state.pref_current = current - 1
                st.rerun()
    with c_save:
        btn_label = "Lưu & Tiếp tục" if current < total - 1 else "Lưu & Hoàn thành"
        if st.button(btn_label, key=f"pref_save_{current}", type="primary", use_container_width=True):
            label_to_key = {_DISPLAY_NAMES[sk]: sk for sk in _DISPLAY_ORDER}
            a["best_system"] = label_to_key[best_label]
            a["note"] = note
            a["done"] = True
            _save_file(questions)

            # Advance to next undone
            if current < total - 1:
                st.session_state.pref_current = current + 1
            else:
                st.session_state.pref_current = _first_undone()
            st.rerun()
    with c_next:
        if current < total - 1:
            if st.button("Bỏ qua", key=f"pref_skip_{current}", use_container_width=True):
                st.session_state.pref_current = current + 1
                st.rerun()


# ── Completion screen ─────────────────────────────────────────────────────────

def _render_completion(questions: list[dict[str, Any]]) -> None:
    total = len(questions)
    done = _done_count()

    st.markdown(
        f'<div class="pref-complete-banner">'
        f'  <div class="pref-complete-title">Hoàn thành!</div>'
        f'  <div class="pref-complete-sub">Cảm ơn <strong>{st.session_state.pref_user}</strong> đã tham gia đánh giá</div>'
        f'  <div class="pref-complete-meta">'
        f'    <span class="pref-complete-chip">{done}/{total} câu hoàn thành</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Vote tally
    votes: dict[str, int] = {sk: 0 for sk in _DISPLAY_ORDER}
    for a in st.session_state.pref_answers:
        if a["done"] and a["best_system"] in votes:
            votes[a["best_system"]] += 1

    st.markdown("#### Kết quả tổng hợp")
    cards_html = '<div class="pref-score-grid">'
    for sk in _DISPLAY_ORDER:
        color = _DISPLAY_COLORS[sk]
        icon = _DISPLAY_ICONS[sk]
        name = _DISPLAY_NAMES[sk]
        v = votes[sk]
        pct = (v / total * 100) if total > 0 else 0
        cards_html += (
            f'<div class="pref-score-card" style="border-top:3px solid {color}">'
            f'  <div class="pref-score-sys">{icon} {name}</div>'
            f'  <div class="pref-score-val" style="color:{color}">{v}</div>'
            f'  <div class="pref-score-label">/ {total} ({pct:.0f}%)</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Detail table
    with st.expander("Xem chi tiết từng câu"):
        import pandas as pd
        rows = []
        for i, q in enumerate(questions):
            a = st.session_state.pref_answers[i]
            _, cat_label = _cat_badge(q["category"])
            rows.append({
                "Câu": i + 1,
                "Mức độ": cat_label,
                "Câu hỏi": q["question"][:60] + ("..." if len(q["question"]) > 60 else ""),
                "Lựa chọn": _DISPLAY_NAMES.get(a["best_system"], "-"),
                "Ghi chú": a["note"][:50] if a["note"] else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Email submission
    st.markdown("---")
    st.markdown(
        "<div style='font-size:1.05rem;font-weight:700;color:#1a2a5e;margin-bottom:8px'>"
        "Nộp kết quả cho nghiên cứu viên</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.pref_submitted:
        st.success("Kết quả đã được nộp thành công! Cảm ơn bạn rất nhiều.")
    elif not settings.gmail.is_configured:
        st.warning("Email chưa được cấu hình — vui lòng liên hệ quản trị viên.")
    else:
        st.caption("Nhấn nút bên dưới để gửi toàn bộ kết quả đánh giá về cho nghiên cứu viên.")
        if st.button("Nộp kết quả", type="primary", use_container_width=True):
            path = _save_file(questions)
            data = json.loads(path.read_text(encoding="utf-8"))
            with st.spinner("Đang gửi email..."):
                try:
                    from src.utils.email_sender import send_experiment_result
                    send_experiment_result(
                        user=st.session_state.pref_user,
                        data=data,
                        sender=settings.gmail.sender,
                        app_password=settings.gmail.app_password,
                        receiver=settings.gmail.receiver,
                    )
                    st.session_state.pref_submitted = True
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"{exc}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


def _md_safe(text: str) -> str:
    """Convert markdown-ish text to safe HTML for the answer box."""
    import re
    # Escape HTML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold: **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Newlines → <br>
    text = text.replace("\n", "<br>")
    return text
