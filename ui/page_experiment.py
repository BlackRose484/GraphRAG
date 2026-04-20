"""
🧪 Experiment page — thử nghiệm người dùng với bộ 21 câu hỏi.

Luồng: Nhập tên → Chọn câu (radio) → Bấm Hỏi → Hiện kết quả 3 cột
     → Đánh giá → Lưu & Next → Chọn câu tiếp theo.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.core.settings import settings
from ui.components import render_retrieval_detail as _render_retrieval_detail
from ui.page_compare import _run_all, _SystemResult, _COLUMN_META

# ── Question Bank ────────────────────────────────────────────────────────────

_QUESTIONS: dict[str, list[dict[str, str]]] = {
    "Dạng 1 — Tra cứu trực tiếp (Dễ)": [
        {"id": "CASE_001", "q": "Mô tả đặc điểm nhân vật Thị Kính?"},
        {"id": "CASE_003", "q": "Mô tả đặc điểm nhân vật Súy Vân?"},
        {"id": "CASE_012", "q": "Ai đóng vai Thị Màu trong vở Quan Âm Thị Kính?"},
        {"id": "CASE_017", "q": "Ai đóng vai Từ Thức trong vở Từ Thức?"},
        {"id": "CASE_022", "q": "Bá Dũng đóng vai gì trong vở Kim Nham?"},
        {"id": "CASE_026", "q": "Vở Quan Âm Thị Kính có những trích đoạn nào?"},
        {"id": "CASE_029", "q": 'Trích đoạn "Vu quy" trong vở Quan Âm Thị Kính miêu tả sự kiện gì?'},
    ],
    "Dạng 2 — Tổng hợp & Quan hệ (Trung bình)": [
        {"id": "CASE_036", "q": "Tất cả những diễn viên nào đã đóng vai Thị Màu?"},
        {"id": "CASE_045", "q": "Liệt kê tất cả nhân vật trong vở Quan Âm Thị Kính?"},
        {"id": "CASE_051", "q": "Những diễn viên nào đã tham gia đồng thời cả vở Quan Âm Thị Kính và Kim Nham?"},
        {"id": "CASE_052", "q": "Mối quan hệ giữa Thị Kính và Thiện Sỹ là gì?"},
        {"id": "CASE_057", "q": "Trần Phương tác động đến Súy Vân như thế nào?"},
        {"id": "CASE_060", "q": "Thị Kính bị Sùng Bà vu oan như thế nào?"},
        {"id": "CASE_063", "q": "Diễn viên An Chinh đã đóng những vai nào?"},
    ],
    "Dạng 3 — Phân tích & So sánh (Khó)": [
        {"id": "CASE_078", "q": "Hãy liệt kê những vở chèo cổ tiêu biểu?"},
        {"id": "CASE_082", "q": "So sánh chủ đề của vở Quan Âm Thị Kính và Kim Nham?"},
        {"id": "CASE_083", "q": "So sánh nhân vật nữ chính của các vở chèo cổ?"},
        {"id": "CASE_090", "q": "Nhân vật loại Đào trong chèo là gì? Có những ai là đại diện tiêu biểu?"},
        {"id": "CASE_093", "q": "Phân tích hình tượng người phụ nữ hy sinh trong các vở chèo cổ?"},
        {"id": "CASE_094", "q": "So sánh số phận bi kịch của Súy Vân và Thị Kính?"},
        {"id": "CASE_097", "q": "Liệt kê các mối quan hệ vợ-chồng trong các vở chèo cổ và kết cục của họ?"},
    ],
}

_CATEGORY_KEYS = list(_QUESTIONS.keys())
_STEP_LABELS   = _CATEGORY_KEYS + ["Câu hỏi tự do"]
_RESULTS_DIR   = Path(__file__).resolve().parents[1] / "benchmark" / "results" / "user_studies"

# ── Visual constants ─────────────────────────────────────────────────────────

_SYS_COLOR = {
    "graphrag": "#2ecc71",
    "rag":      "#3498db",
    "chat":     "#e67e22",
}
_SYS_ICON = {
    "graphrag": "🕸️",
    "rag":      "📚",
    "chat":     "🤖",
}
_DIFF_META = [
    ("#27ae60", "🟢 Dễ"),
    ("#f39c12", "🟡 Trung bình"),
    ("#e74c3c", "🔴 Khó"),
    ("#9b59b6", "🟣 Tự do"),
]
_STEP_SHORT = ["Dễ", "Trung bình", "Khó", "Tự do"]

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Hero ── */
.exp-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    border-radius: 16px; padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem; text-align: center;
}
.exp-hero-title {
    font-size: 1.9rem; font-weight: 800; color: #ffffff;
    margin: 0 0 0.4rem; letter-spacing: -0.3px;
}
.exp-hero-sub {
    font-size: 1rem; color: rgba(255,255,255,0.75); margin: 0;
}

/* ── System cards on gate page ── */
.sys-cards { display: flex; gap: 12px; margin: 1.4rem 0 0.5rem; }
.sys-card {
    flex: 1; border-radius: 12px; padding: 1.1rem 0.8rem;
    text-align: center; color: white;
}
.sys-card-icon  { font-size: 2rem; margin-bottom: 6px; }
.sys-card-name  { font-weight: 700; font-size: 0.95rem; }
.sys-card-desc  { font-size: 0.75rem; opacity: 0.85; margin-top: 4px; line-height: 1.3; }

/* ── Hint pills row ── */
.hint-row {
    display: flex; gap: 18px; justify-content: center;
    margin-top: 1rem; flex-wrap: wrap;
}
.hint-pill {
    background: rgba(255,255,255,0.12); border-radius: 20px;
    padding: 4px 14px; font-size: 0.8rem; color: rgba(255,255,255,0.88);
}

/* ── Stepper ── */
.stepper-wrap {
    display: flex; align-items: flex-start; padding: 0.4rem 0.2rem 1rem;
    gap: 4px; margin-bottom: 0.2rem;
}
.step-col  { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; min-width: 64px; }
.step-circle {
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1rem; border: 2px solid;
}
.step-circle.done    { background:#27ae60; border-color:#27ae60; color:white; }
.step-circle.active  { background:#2980b9; border-color:#2980b9; color:white;
                        box-shadow: 0 0 0 5px rgba(41,128,185,0.22); }
.step-circle.pending { background:#f0f2f6; border-color:#ccc; color:#bbb; }
.step-lbl { font-size:0.68rem; color:#777; margin-top:5px; text-align:center; }
.step-line { flex:1; height:2px; margin-top:19px; min-width:16px; border-radius:2px; }

/* ── Step header card ── */
.step-header {
    background: linear-gradient(90deg, #f0f4ff 0%, #fafbff 100%);
    border: 1px solid #dde6f5; border-left: 4px solid;
    border-radius: 0 10px 10px 0; padding: 0.8rem 1.2rem;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
}
.step-header-num  { font-size: 1.5rem; font-weight: 800; color: #2c3e7f; }
.step-header-body { flex: 1; }
.step-header-title{ font-size: 1rem; font-weight: 700; color: #1a2a5e; margin: 0; }
.step-header-sub  { font-size: 0.78rem; color: #888; margin: 2px 0 0; }
.diff-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; color: white;
}

/* ── Column header ── */
.col-hdr {
    border-radius: 8px; padding: 9px 14px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 8px;
}
.col-hdr-dot   { width:12px; height:12px; border-radius:50%; flex-shrink:0; }
.col-hdr-title { font-weight: 700; font-size: 1rem; color: white; }
.col-hdr-sub   { font-size: 0.72rem; color: rgba(255,255,255,0.8); margin-left: auto; }

/* ── Rating section ── */
.rating-wrap {
    background: #f8faff; border: 1px solid #e0e8f5;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-top: 1rem;
}
.rating-title {
    font-size: 1rem; font-weight: 700; color: #1a2a5e;
    margin-bottom: 1rem; border-bottom: 2px solid #e0e8f5; padding-bottom: 0.5rem;
}
.rating-col-hdr {
    border-radius: 6px; padding: 6px 12px; margin-bottom: 8px;
    font-weight: 700; font-size: 0.9rem; color: white;
    display: flex; align-items: center; gap: 6px;
}

/* ── Done step expander summary ── */
.done-summary {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px;
}
.done-pill {
    border-radius: 16px; padding: 2px 10px; font-size: 0.75rem;
    font-weight: 600; color: white;
}

/* ── Completion banner ── */
.complete-banner {
    background: linear-gradient(135deg, #0d7a45 0%, #27ae60 60%, #2ecc71 100%);
    border-radius: 16px; padding: 2rem 2rem 1.6rem;
    text-align: center; margin-bottom: 1.4rem;
}
.complete-title { font-size: 1.8rem; font-weight: 800; color: white; margin: 0 0 0.3rem; }
.complete-sub   { font-size: 1rem; color: rgba(255,255,255,0.88); margin: 0; }
.complete-meta  { margin-top: 1rem; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
.complete-chip  {
    background: rgba(255,255,255,0.18); border-radius: 20px;
    padding: 4px 16px; font-size: 0.82rem; color: white;
}

/* ── Score summary ── */
.score-grid { display: flex; gap: 10px; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
.score-card {
    flex: 1; min-width: 110px; background: white;
    border: 1px solid #e5e9f0; border-radius: 10px;
    padding: 0.8rem 0.6rem; text-align: center;
}
.score-sys   { font-size: 0.72rem; color: #888; font-weight: 600; text-transform: uppercase; }
.score-val   { font-size: 1.8rem; font-weight: 800; color: #1a2a5e; line-height: 1.1; }
.score-label { font-size: 0.68rem; color: #aaa; }
</style>
"""


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _stepper_html(done_count: int, current: int) -> str:
    parts: list[str] = []
    for i in range(4):
        if i < done_count:
            cls, inner = "done", "✓"
        elif i == current and current < 4:
            cls, inner = "active", str(i + 1)
        else:
            cls, inner = "pending", str(i + 1)

        parts.append(
            f'<div class="step-col">'
            f'  <div class="step-circle {cls}">{inner}</div>'
            f'  <div class="step-lbl">{_STEP_SHORT[i]}</div>'
            f'</div>'
        )
        if i < 3:
            line_bg = "#27ae60" if i < done_count else "#e0e4ea"
            parts.append(f'<div class="step-line" style="background:{line_bg}"></div>')

    return f'<div class="stepper-wrap">{"".join(parts)}</div>'


def _step_header_html(current: int) -> str:
    diff_color, diff_label = _DIFF_META[current]
    category = _STEP_LABELS[current] if current < len(_STEP_LABELS) else ""
    return (
        f'<div class="step-header" style="border-left-color:{diff_color}">'
        f'  <div class="step-header-num">{current + 1}</div>'
        f'  <div class="step-header-body">'
        f'    <div class="step-header-title">{category}</div>'
        f'    <div class="step-header-sub">Bước {current + 1} / 4</div>'
        f'  </div>'
        f'  <span class="diff-badge" style="background:{diff_color}">{diff_label}</span>'
        f'</div>'
    )


def _col_header_html(key: str, elapsed: float | None = None) -> str:
    color = _SYS_COLOR[key]
    icon  = _SYS_ICON[key]
    title = _COLUMN_META[key]["title"]
    sub   = f"⏱ {elapsed:.1f}s" if elapsed is not None else ""
    return (
        f'<div class="col-hdr" style="background:{color}">'
        f'  <span style="font-size:1.2rem">{icon}</span>'
        f'  <span class="col-hdr-title">{title}</span>'
        f'  <span class="col-hdr-sub">{sub}</span>'
        f'</div>'
    )


def _rating_col_hdr_html(key: str) -> str:
    color = _SYS_COLOR[key]
    icon  = _SYS_ICON[key]
    title = _COLUMN_META[key]["title"]
    return (
        f'<div class="rating-col-hdr" style="background:{color}">'
        f'  {icon} {title}'
        f'</div>'
    )


# ── Core helpers (logic unchanged) ───────────────────────────────────────────

def _init_state() -> None:
    if "exp_user" not in st.session_state:
        st.session_state.exp_user = ""
    if "exp_steps" not in st.session_state:
        st.session_state.exp_steps = [
            {"done": False, "question": "", "case_id": "", "results": {}, "rating": {}}
            for _ in range(4)
        ]
    if "exp_current" not in st.session_state:
        st.session_state.exp_current = 0
    if "exp_awaiting_rating" not in st.session_state:
        st.session_state.exp_awaiting_rating = False
    if "exp_submitted" not in st.session_state:
        st.session_state.exp_submitted = False


def _current_step() -> int:
    for i, s in enumerate(st.session_state.exp_steps):
        if not s["done"]:
            return i
    return 4


def _save_file() -> Path:
    user = st.session_state.exp_user or "anonymous"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / f"experiment_{user}.json"
    data: dict[str, Any] = {
        "user": user,
        "last_updated": datetime.now().isoformat(),
        "steps": [],
    }
    for i, step in enumerate(st.session_state.exp_steps):
        if not step["done"]:
            continue
        sd: dict[str, Any] = {
            "step": i + 1, "category": _STEP_LABELS[i],
            "case_id": step.get("case_id", ""),
            "question": step.get("question", ""),
            "rating": step.get("rating", {}),
            "results": {},
        }
        for sk in ("graphrag", "rag", "chat"):
            r: _SystemResult | None = step.get("results", {}).get(sk)
            if r:
                sd["results"][sk] = {
                    "answer": r.answer, "elapsed": round(r.elapsed, 2),
                    "error": r.error, "metadata": r.metadata,
                }
        data["steps"].append(sd)
    data["total_completed"] = len(data["steps"])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Render helpers ────────────────────────────────────────────────────────────

def _render_columns(results: dict[str, _SystemResult], kp: str, expander_ok: bool = True) -> None:
    col_g, col_r, col_c = st.columns(3)
    for col, name in [(col_g, "graphrag"), (col_r, "rag"), (col_c, "chat")]:
        r = results.get(name)
        if r is None:
            continue
        elapsed = r.elapsed if not r.error else None
        col.markdown(_col_header_html(name, elapsed), unsafe_allow_html=True)
        if r.error:
            col.error(f"❌ {r.error[:200]}")
            continue
        col.markdown(r.answer)
        if r.metadata:
            pills = []
            for k, lbl in [("num_nodes", "🗂"), ("num_triplets", "🔗"), ("num_chunks", "📄")]:
                if k in r.metadata:
                    pills.append(f"{lbl} {r.metadata[k]}")
            if pills:
                col.caption("  ·  ".join(pills))
        if expander_ok and name == "graphrag" and r.retrieval_detail:
            with col.expander("📊 Chi tiết retrieval", expanded=False):
                _render_retrieval_detail(r.retrieval_detail, key_prefix=kp, compact=True)


def _render_rating(step_idx: int) -> None:
    st.markdown(
        '<div class="rating-wrap">'
        '<div class="rating-title">📝 Đánh giá câu trả lời — thang điểm 1 (kém) → 5 (xuất sắc)</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    systems = [("GraphRAG", "graphrag"), ("RAG", "rag"), ("LLM", "chat")]
    cols = st.columns(3)
    ratings: dict[str, dict[str, int]] = {}
    for col, (label, key) in zip(cols, systems):
        with col:
            col.markdown(_rating_col_hdr_html(key), unsafe_allow_html=True)
            a = st.slider("Chính xác",  1, 5, 3, key=f"exp_r_acc_{step_idx}_{key}")
            b = st.slider("Đầy đủ",     1, 5, 3, key=f"exp_r_comp_{step_idx}_{key}")
            c = st.slider("Tự nhiên",   1, 5, 3, key=f"exp_r_nat_{step_idx}_{key}")
            ratings[key] = {"accuracy": a, "completeness": b, "naturalness": c}

    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        best = st.selectbox(
            "🏆 Hệ thống tốt nhất?",
            ["GraphRAG", "RAG", "LLM"],
            key=f"exp_best_{step_idx}",
        )
    with c2:
        note = st.text_area(
            "Ghi chú (tùy chọn)",
            key=f"exp_note_{step_idx}",
            placeholder="Nhận xét thêm về chất lượng câu trả lời...",
            height=80,
        )

    if st.button("💾 Lưu đánh giá & Tiếp tục →", key=f"exp_save_{step_idx}", type="primary"):
        best_map = {"GraphRAG": "graphrag", "RAG": "rag", "LLM": "chat"}
        step = st.session_state.exp_steps[step_idx]
        step["rating"] = {"per_system": ratings, "best_system": best_map[best], "note": note}
        step["done"] = True
        st.session_state.exp_awaiting_rating = False
        st.session_state.exp_current = _current_step()
        _save_file()
        st.rerun()


# ── Main render ──────────────────────────────────────────────────────────────

def render() -> None:
    _init_state()

    # Inject CSS
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    done_count = sum(1 for s in st.session_state.exp_steps if s["done"])
    st.sidebar.markdown("### 🧪 Tiến độ thử nghiệm")
    st.sidebar.progress(done_count / 4, text=f"{done_count} / 4 câu hoàn thành")
    st.sidebar.markdown("")
    for i, label in enumerate(_STEP_LABELS):
        diff_color, diff_label = _DIFF_META[i]
        icon = "✅" if st.session_state.exp_steps[i]["done"] else "🔲"
        st.sidebar.markdown(
            f"{icon} &nbsp;"
            f"<span style='font-size:0.8em;background:{diff_color}22;"
            f"border:1px solid {diff_color}55;border-radius:4px;"
            f"padding:1px 6px;color:{diff_color};font-weight:600'>"
            f"{_STEP_SHORT[i]}</span> "
            f"<span style='font-size:0.85em'>{label.split('—')[0].strip()}</span>",
            unsafe_allow_html=True,
        )
    if done_count > 0:
        st.sidebar.markdown("---")
        path = _save_file()
        with open(path, "r", encoding="utf-8") as f:
            st.sidebar.download_button(
                "⬇️ Tải kết quả JSON", f.read(),
                file_name=path.name, mime="application/json",
            )
    if st.sidebar.button("🔄 Làm lại từ đầu"):
        for k in [k for k in st.session_state if k.startswith("exp_")]:
            del st.session_state[k]
        st.rerun()

    # ── Gate: nhập tên ───────────────────────────────────────────────────────
    if not st.session_state.exp_user:
        st.markdown(
            '<div class="exp-hero">'
            '  <div class="exp-hero-title">🎭 Thử nghiệm Hệ thống Hỏi–Đáp</div>'
            '  <div class="exp-hero-sub">So sánh ba phương pháp trả lời câu hỏi về Nghệ thuật Chèo Việt Nam</div>'
            '  <div class="sys-cards">'
            '    <div class="sys-card" style="background:rgba(46,204,113,0.85)">'
            '      <div class="sys-card-icon">🕸️</div>'
            '      <div class="sys-card-name">GraphRAG</div>'
            '      <div class="sys-card-desc">Knowledge Graph<br>+ Mô hình ngôn ngữ</div>'
            '    </div>'
            '    <div class="sys-card" style="background:rgba(52,152,219,0.85)">'
            '      <div class="sys-card-icon">📚</div>'
            '      <div class="sys-card-name">RAG</div>'
            '      <div class="sys-card-desc">Tìm kiếm ngữ nghĩa<br>+ Tổng hợp văn bản</div>'
            '    </div>'
            '    <div class="sys-card" style="background:rgba(230,126,34,0.85)">'
            '      <div class="sys-card-icon">🤖</div>'
            '      <div class="sys-card-name">LLM</div>'
            '      <div class="sys-card-desc">Mô hình ngôn ngữ<br>trả lời trực tiếp</div>'
            '    </div>'
            '  </div>'
            '  <div class="hint-row">'
            '    <span class="hint-pill">📋 4 câu hỏi</span>'
            '    <span class="hint-pill">⏱ ~20 phút</span>'
            '    <span class="hint-pill">💾 Tự động lưu</span>'
            '    <span class="hint-pill">🔒 Ẩn danh</span>'
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
            with st.form("gate_form"):
                name = st.text_input(
                    "Tên", placeholder="Ví dụ: Nguyễn Văn A",
                    key="gate_name", label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "🚀 Bắt đầu thử nghiệm",
                    type="primary", use_container_width=True,
                )
            if submitted:
                if name.strip():
                    st.session_state.exp_user = name.strip()
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập tên trước khi bắt đầu.")

        return

    # ── Header & stepper (always visible after login) ─────────────────────────
    st.markdown(
        f"<div style='font-size:0.9rem;color:#888;margin-bottom:4px'>"
        f"Xin chào, <strong>{st.session_state.exp_user}</strong> 👋</div>",
        unsafe_allow_html=True,
    )
    current = _current_step()
    st.markdown(_stepper_html(done_count, current), unsafe_allow_html=True)

    # ── Các bước đã hoàn thành ────────────────────────────────────────────────
    for i, step in enumerate(st.session_state.exp_steps):
        if not step["done"]:
            continue
        rat      = step.get("rating", {})
        best_sys = rat.get("best_system", "")
        best_color = _SYS_COLOR.get(best_sys, "#999")
        diff_color, diff_label = _DIFF_META[i]

        with st.expander(
            f"✅  Bước {i+1}  ·  {_STEP_SHORT[i]}  —  \"{step['question'][:55]}...\"",
            expanded=False,
        ):
            # Mini rating summary pills
            if rat:
                pills_html = ""
                for sk, sl in [("graphrag", "GraphRAG"), ("rag", "RAG"), ("chat", "LLM")]:
                    ps = rat.get("per_system", {}).get(sk, {})
                    avg = (
                        ps.get("accuracy", 0) + ps.get("completeness", 0) + ps.get("naturalness", 0)
                    ) / 3
                    c = _SYS_COLOR[sk]
                    star = " ⭐" if best_sys == sk else ""
                    pills_html += (
                        f"<span class='done-pill' style='background:{c}'>"
                        f"{sl}: {avg:.1f}{star}</span>"
                    )
                st.markdown(
                    f'<div class="done-summary">{pills_html}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            with st.chat_message("user"):
                st.markdown(step["question"])
            _render_columns(step["results"], kp=f"exp_done_{i}_", expander_ok=False)

    # ── Tất cả xong ──────────────────────────────────────────────────────────
    if current >= 4:
        st.markdown(
            f'<div class="complete-banner">'
            f'  <div class="complete-title">🎉 Hoàn thành xuất sắc!</div>'
            f'  <div class="complete-sub">Cảm ơn <strong>{st.session_state.exp_user}</strong> đã tham gia thử nghiệm</div>'
            f'  <div class="complete-meta">'
            f'    <span class="complete-chip">✅ 4/4 câu hoàn thành</span>'
            f'    <span class="complete-chip">📊 12 đánh giá đã ghi nhận</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Score summary cards per system
        st.markdown("#### 📊 Tổng điểm trung bình theo hệ thống")
        sys_totals: dict[str, list[float]] = {"graphrag": [], "rag": [], "chat": []}
        best_votes: dict[str, int]          = {"graphrag": 0, "rag": 0, "chat": 0}
        for step in st.session_state.exp_steps:
            rat = step.get("rating", {})
            for sk in ("graphrag", "rag", "chat"):
                ps = rat.get("per_system", {}).get(sk, {})
                if ps:
                    sys_totals[sk].append(
                        (ps.get("accuracy", 0) + ps.get("completeness", 0) + ps.get("naturalness", 0)) / 3
                    )
            bv = rat.get("best_system", "")
            if bv in best_votes:
                best_votes[bv] += 1

        cards_html = '<div class="score-grid">'
        for sk, sl, icon in [("graphrag", "GraphRAG", "🕸️"), ("rag", "RAG", "📚"), ("chat", "LLM", "🤖")]:
            vals = sys_totals[sk]
            avg  = sum(vals) / len(vals) if vals else 0
            votes = best_votes[sk]
            color = _SYS_COLOR[sk]
            cards_html += (
                f'<div class="score-card" style="border-top:3px solid {color}">'
                f'  <div class="score-sys">{icon} {sl}</div>'
                f'  <div class="score-val" style="color:{color}">{avg:.2f}</div>'
                f'  <div class="score-label">/ 5.00 · 🏆 {votes}×</div>'
                f'</div>'
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        # Full detail table
        with st.expander("📋 Xem bảng chi tiết"):
            rows = []
            for i, step in enumerate(st.session_state.exp_steps):
                rat = step.get("rating", {})
                ps  = rat.get("per_system", {})
                for sk, sl in [("graphrag", "GraphRAG"), ("rag", "RAG"), ("chat", "LLM")]:
                    s = ps.get(sk, {})
                    r = step.get("results", {}).get(sk)
                    rows.append({
                        "Bước": i + 1, "Hệ thống": sl,
                        "Chính xác": s.get("accuracy", "-"),
                        "Đầy đủ": s.get("completeness", "-"),
                        "Tự nhiên": s.get("naturalness", "-"),
                        "Thời gian": f"{r.elapsed:.1f}s" if r else "-",
                        "Tốt nhất": "⭐" if rat.get("best_system") == sk else "",
                    })
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Nộp kết quả
        st.markdown("---")
        st.markdown(
            "<div style='font-size:1.05rem;font-weight:700;color:#1a2a5e;margin-bottom:8px'>"
            "📧 Nộp kết quả cho nghiên cứu viên</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.exp_submitted:
            st.success("✅ Kết quả đã được nộp thành công! Cảm ơn bạn rất nhiều.")
        elif not settings.gmail.is_configured:
            st.warning("⚠️ Email chưa được cấu hình — vui lòng liên hệ quản trị viên.")
        else:
            st.caption("Nhấn nút bên dưới để gửi toàn bộ kết quả đánh giá về cho nghiên cứu viên.")
            if st.button(
                "📧 Nộp kết quả",
                type="primary", use_container_width=True,
            ):
                path = _save_file()
                data = json.loads(path.read_text(encoding="utf-8"))
                with st.spinner("Đang gửi email..."):
                    try:
                        from src.utils.email_sender import send_experiment_result
                        send_experiment_result(
                            user=st.session_state.exp_user,
                            data=data,
                            sender=settings.gmail.sender,
                            app_password=settings.gmail.app_password,
                            receiver=settings.gmail.receiver,
                        )
                        st.session_state.exp_submitted = True
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(f"❌ {exc}")
        return

    # ── Bước hiện tại ─────────────────────────────────────────────────────────
    st.markdown(_step_header_html(current), unsafe_allow_html=True)

    step = st.session_state.exp_steps[current]

    # Nếu đã có kết quả chờ đánh giá
    if st.session_state.exp_awaiting_rating and step["results"]:
        with st.chat_message("user"):
            st.markdown(step["question"])
        _render_columns(step["results"], kp=f"exp_cur_{current}_")
        _render_rating(current)
        return

    # Chọn câu hỏi
    if current < 3:
        cat_key   = _CATEGORY_KEYS[current]
        questions = _QUESTIONS[cat_key]
        diff_color, diff_label = _DIFF_META[current]
        st.markdown(
            f"<div style='background:{diff_color}15;border:1px solid {diff_color}40;"
            f"border-radius:8px;padding:8px 14px;font-size:0.88rem;margin-bottom:10px'>"
            f"📋 Chọn <strong>1 câu hỏi</strong> từ danh sách — đây là nhóm <strong>{diff_label}</strong></div>",
            unsafe_allow_html=True,
        )
        with st.form(f"question_form_{current}"):
            choice_idx = st.radio(
                "Chọn câu hỏi:",
                range(len(questions)),
                format_func=lambda i: f"{questions[i]['id']}: {questions[i]['q']}",
                key=f"exp_choice_{current}",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("🚀 Hỏi câu này", type="primary")
        if submitted:
            chosen = questions[choice_idx]
            step["question"] = chosen["q"]
            step["case_id"]  = chosen["id"]
            with st.spinner("⏳ Đang chạy song song 3 hệ thống — GraphRAG · RAG · LLM..."):
                step["results"] = _run_all(chosen["q"])
            st.session_state.exp_awaiting_rating = True
            st.rerun()
    else:
        st.markdown(
            "<div style='background:#9b59b615;border:1px solid #9b59b640;"
            "border-radius:8px;padding:8px 14px;font-size:0.88rem;margin-bottom:10px'>"
            "✏️ Hãy đặt <strong>1 câu hỏi tự do</strong> bất kỳ về nghệ thuật Chèo</div>",
            unsafe_allow_html=True,
        )
        with st.form("free_form"):
            free_q = st.text_input(
                "Câu hỏi của bạn:",
                key="exp_free_q",
                placeholder="Ví dụ: Vở chèo nào có kết thúc có hậu?",
            )
            submitted = st.form_submit_button("🚀 Hỏi câu này", type="primary")
        if submitted:
            if free_q.strip():
                step["question"] = free_q.strip()
                step["case_id"]  = "FREE"
                with st.spinner("⏳ Đang chạy song song 3 hệ thống — GraphRAG · RAG · LLM..."):
                    step["results"] = _run_all(free_q.strip())
                st.session_state.exp_awaiting_rating = True
                st.rerun()
            else:
                st.warning("Vui lòng nhập câu hỏi trước khi gửi.")
