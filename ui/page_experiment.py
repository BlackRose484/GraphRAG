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
        {"id": "CASE_063", "q": "Diễn viên An Chinh đã đóng bao nhiêu vai khác nhau trong KG?"},
    ],
    "Dạng 3 — Phân tích & So sánh (Khó)": [
        {"id": "CASE_078", "q": "Những vở chèo nào có trong KG? Liệt kê đầy đủ?"},
        {"id": "CASE_082", "q": "So sánh chủ đề của vở Quan Âm Thị Kính và Kim Nham?"},
        {"id": "CASE_083", "q": "So sánh nhân vật nữ chính của các vở trong KG?"},
        {"id": "CASE_090", "q": "Nhân vật loại Đào trong chèo là gì? Ai là đại diện trong KG?"},
        {"id": "CASE_093", "q": "Phân tích hình tượng người phụ nữ hy sinh trong KG?"},
        {"id": "CASE_094", "q": "So sánh số phận bi kịch của Súy Vân và Thị Kính?"},
        {"id": "CASE_097", "q": "Liệt kê tất cả mối quan hệ vợ-chồng trong KG và kết cục của họ?"},
    ],
}

_CATEGORY_KEYS = list(_QUESTIONS.keys())
_STEP_LABELS = _CATEGORY_KEYS + ["Câu hỏi tự do"]
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "results"


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _render_columns(results: dict[str, _SystemResult], kp: str, expander_ok: bool = True) -> None:
    col_g, col_r, col_c = st.columns(3)
    for col, name in [(col_g, "graphrag"), (col_r, "rag"), (col_c, "chat")]:
        meta = _COLUMN_META[name]
        r = results.get(name)
        if r is None:
            continue
        col.markdown(
            f"<div style='border-bottom:3px solid {meta['color']};padding-bottom:4px;"
            f"margin-bottom:8px;font-weight:700;font-size:1.1rem'>{meta['title']}</div>",
            unsafe_allow_html=True,
        )
        if r.error:
            col.error(f"❌ {r.error[:200]}")
            col.caption(f"⏱ {r.elapsed:.1f}s")
            continue
        col.markdown(r.answer)
        col.caption(f"⏱ **{r.elapsed:.1f}s**")
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
    st.markdown("---")
    st.markdown("#### 📝 Đánh giá câu trả lời")
    systems = [("GraphRAG", "graphrag"), ("RAG", "rag"), ("LLM", "chat")]
    cols = st.columns(3)
    ratings: dict[str, dict[str, int]] = {}
    for col, (label, key) in zip(cols, systems):
        with col:
            st.markdown(f"**{label}**")
            a = st.slider("Chính xác", 1, 5, 3, key=f"exp_r_acc_{step_idx}_{key}")
            b = st.slider("Đầy đủ", 1, 5, 3, key=f"exp_r_comp_{step_idx}_{key}")
            c = st.slider("Tự nhiên", 1, 5, 3, key=f"exp_r_nat_{step_idx}_{key}")
            ratings[key] = {"accuracy": a, "completeness": b, "naturalness": c}

    best = st.selectbox("🏆 Hệ thống tốt nhất?", ["GraphRAG", "RAG", "LLM"], key=f"exp_best_{step_idx}")
    note = st.text_area("Ghi chú (tùy chọn)", key=f"exp_note_{step_idx}", placeholder="Nhận xét thêm...")

    if st.button("💾 Lưu đánh giá & Tiếp tục", key=f"exp_save_{step_idx}", type="primary"):
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

    st.title("🧪 Thử nghiệm người dùng")
    st.caption("Chọn câu hỏi → Xem kết quả 3 hệ thống → Đánh giá → Tiếp tục")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🧪 Thử nghiệm")
    done_count = sum(1 for s in st.session_state.exp_steps if s["done"])
    st.sidebar.progress(done_count / 4, text=f"{done_count}/4 câu hoàn thành")
    for i, label in enumerate(_STEP_LABELS):
        icon = "✅" if st.session_state.exp_steps[i]["done"] else "🔲"
        st.sidebar.markdown(f"{icon} {label}")
    if done_count > 0:
        st.sidebar.markdown("---")
        path = _save_file()
        with open(path, "r", encoding="utf-8") as f:
            st.sidebar.download_button("⬇️ Tải JSON", f.read(), file_name=path.name, mime="application/json")
    if st.sidebar.button("🔄 Làm lại từ đầu"):
        for k in [k for k in st.session_state if k.startswith("exp_")]:
            del st.session_state[k]
        st.rerun()

    # ── Gate: nhập tên ───────────────────────────────────────────────────────
    if not st.session_state.exp_user:
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("#### Vui lòng nhập tên để bắt đầu")
            name = st.text_input("Tên", placeholder="Ví dụ: Nguyễn Văn A", key="gate_name", label_visibility="collapsed")
            if st.button("🚀 Bắt đầu", type="primary", use_container_width=True, disabled=not name):
                st.session_state.exp_user = name.strip()
                st.rerun()
            st.info(
                "📋 **Hướng dẫn:**\n"
                "- Chọn **1 câu/dạng** (Dễ, Trung bình, Khó) + **1 câu tự do** = 4 câu\n"
                "- Mỗi câu chạy 3 hệ thống song song để bạn so sánh\n"
                "- Kết quả **tự động lưu** sau mỗi đánh giá"
            )
        return

    # ── Hiện các bước đã hoàn thành ──────────────────────────────────────────
    for i, step in enumerate(st.session_state.exp_steps):
        if not step["done"]:
            continue
        with st.expander(f"✅ Bước {i+1}: {_STEP_LABELS[i]} — \"{step['question'][:50]}...\"", expanded=False):
            with st.chat_message("user"):
                st.markdown(step["question"])
            _render_columns(step["results"], kp=f"exp_done_{i}_", expander_ok=False)
            rat = step.get("rating", {})
            if rat:
                st.markdown(f"🏆 **Tốt nhất:** `{rat.get('best_system','')}`")

    # ── Tất cả xong ─────────────────────────────────────────────────────────
    current = _current_step()
    if current >= 4:
        st.success("🎉 Bạn đã hoàn thành tất cả 4 câu hỏi! Cảm ơn bạn đã tham gia.")
        st.markdown("### 📊 Tổng kết")
        rows = []
        for i, step in enumerate(st.session_state.exp_steps):
            rat = step.get("rating", {})
            ps = rat.get("per_system", {})
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
        return

    # ── Bước hiện tại ────────────────────────────────────────────────────────
    st.markdown(f"### Bước {current + 1}/4: {_STEP_LABELS[current]}")
    st.markdown("---")

    step = st.session_state.exp_steps[current]

    # Nếu đã có kết quả chờ đánh giá → hiện kết quả + form
    if st.session_state.exp_awaiting_rating and step["results"]:
        with st.chat_message("user"):
            st.markdown(step["question"])
        _render_columns(step["results"], kp=f"exp_cur_{current}_")
        _render_rating(current)
        return

    # Chọn câu hỏi
    if current < 3:
        cat_key = _CATEGORY_KEYS[current]
        questions = _QUESTIONS[cat_key]
        st.info(f"📋 Chọn **1 câu hỏi** từ danh sách dưới đây:")

        choice_idx = st.radio(
            "Chọn câu hỏi:",
            range(len(questions)),
            format_func=lambda i: f"{questions[i]['id']}: {questions[i]['q']}",
            key=f"exp_choice_{current}",
            label_visibility="collapsed",
        )
        chosen = questions[choice_idx]

        if st.button(f"🚀 Hỏi câu này", type="primary", key=f"exp_ask_{current}"):
            step["question"] = chosen["q"]
            step["case_id"] = chosen["id"]
            with st.spinner("⏳ Đang chạy song song 3 hệ thống — GraphRAG · RAG · LLM..."):
                step["results"] = _run_all(chosen["q"])
            st.session_state.exp_awaiting_rating = True
            st.rerun()
    else:
        st.info("✏️ Hãy đặt **1 câu hỏi tự do** bất kỳ về nghệ thuật Chèo")
        free_q = st.text_input("Câu hỏi của bạn:", key="exp_free_q", placeholder="Ví dụ: Vở chèo nào có kết thúc có hậu?")
        if st.button("🚀 Hỏi câu này", type="primary", key="exp_ask_free", disabled=not free_q):
            step["question"] = free_q
            step["case_id"] = "FREE"
            with st.spinner("⏳ Đang chạy song song 3 hệ thống — GraphRAG · RAG · LLM..."):
                step["results"] = _run_all(free_q)
            st.session_state.exp_awaiting_rating = True
            st.rerun()
