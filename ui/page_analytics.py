"""
📈 Analytics page — thống kê & visualize kết quả khảo sát.

Đọc 2 folder kết quả (đã được fetch về từ Gmail bằng
``scripts.fetch_results``):

    benchmark/results/experiments/   — experiment_*.json (4 câu / người)
    benchmark/results/preferences/   — preference_*.json (21 câu / người)

Cung cấp 4 tab:
    📊 Tổng quan       — KPI + biểu đồ tóm tắt
    📋 Đánh giá ưu tiên — phân tích sâu vote-best-system
    🧪 Thử nghiệm      — rating 3 chiều + latency
    👥 Người tham gia  — drill-down theo user

Có nút "Cập nhật từ Gmail" để chạy fetcher trực tiếp từ UI.
"""
from __future__ import annotations

import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP_DIR  = _PROJECT_ROOT / "benchmark" / "results" / "experiments"
_PREF_DIR = _PROJECT_ROOT / "benchmark" / "results" / "preferences"


def _find_post_survey_csv() -> Path | None:
    """Tìm CSV khảo sát hậu nghiệm — match theo pattern Google Forms export."""
    patterns = [
        "Đánh giá*Form Responses*.csv",
        "*Form Responses 1.csv",
        "post_survey*.csv",
    ]
    for pat in patterns:
        for p in _PROJECT_ROOT.glob(pat):
            return p
        for p in (_PROJECT_ROOT / "benchmark" / "results").glob(pat):
            return p
    return None

# ── Display config (đồng bộ với page_preference) ──────────────────────────────

_SYSTEMS = ["chat", "rag", "graphrag"]
_DISPLAY_NAMES = {
    "chat":     "LLM Only",
    "rag":      "RAG",
    "graphrag": "GraphRAG",
}
_DISPLAY_COLORS = {
    "chat":     "#e67e22",
    "rag":      "#3498db",
    "graphrag": "#2ecc71",
}
_DIMENSIONS = ["accuracy", "completeness", "naturalness"]
_DIMENSION_LABELS = {
    "accuracy":     "Chính xác",
    "completeness": "Đầy đủ",
    "naturalness":  "Tự nhiên",
}

_CAT_ORDER = ["Dạng 1", "Dạng 2", "Dạng 3", "Câu hỏi tự do"]
_CAT_COLORS = {
    "Dạng 1":         "#27ae60",
    "Dạng 2":         "#f39c12",
    "Dạng 3":         "#e74c3c",
    "Câu hỏi tự do": "#9b59b6",
}


# ── Post-survey: column names & canonical multi-choice options ───────────────
_PS_COL_CHEO_KNOWLEDGE = "Trước khi tham gia thử nghiệm, mức độ hiểu biết của bạn về nghệ thuật Chèo là như thế nào?"
_PS_COL_AI_USAGE       = "Tần suất sử dụng các công cụ Trí tuệ nhân tạo (như ChatGPT, Gemini, Copilot...) trong công việc/học tập của bạn?"
_PS_COL_TRUSTED_SYS    = "Xuyên suốt quá trình thử nghiệm, nhìn chung bạn cảm thấy hệ thống (cột) nào đem lại CẢM GIÁC ĐÁNG TIN CẬY nhất?"
_PS_COL_CRITERIA       = 'Khi đưa ra quyết định chọn "Hệ thống tốt nhất" ở mỗi câu hỏi, bạn thường đặt nặng TIÊU CHÍ nào nhất? (Chọn tối đa 2 đáp án)'
_PS_COL_ERRORS         = "Trong các câu trả lời do AI sinh ra mà bạn vừa đọc, bạn có bắt gặp các lỗi nào dưới đây không? (Có thể chọn nhiều đáp án)"
_PS_COL_UI_RATING      = "Bạn đánh giá mức độ thân thiện và dễ sử dụng của Giao diện thử nghiệm (việc bố trí 3 cột cạnh nhau, các nút chấm điểm...) ở mức nào?"
_PS_COL_WAIT_TIME      = "Thời gian chờ đợi hệ thống sinh câu trả lời đối với bạn:"
_PS_COL_IMPRESSION     = "Điều gì làm bạn ấn tượng nhất (hoặc thất vọng nhất) trong buổi thực nghiệm vừa rồi?"
_PS_COL_FEEDBACK       = 'Bạn có góp ý gì để nhóm nghiên cứu tiếp tục hoàn thiện "Hệ thống Hỏi-Đáp về Nghệ thuật Chèo" trong tương lai không?'

_PS_CRITERIA_OPTIONS = [
    "Trả lời đúng sự thật, không bịa đặt thông tin (Accuracy)",
    "Trả lời đầy đủ, không bỏ sót ý quan trọng (Completeness)",
    "Trả lời đúng trọng tâm câu hỏi, không lan man",
    "Cấu trúc trình bày rõ ràng (có khoảng trắng, gạch đầu dòng, in đậm...)",
    "Văn phong mạch lạc, tự nhiên, dễ hiểu (Naturalness)",
    "Ý kiến khác",
]
_PS_CRITERIA_SHORT = {
    "Trả lời đúng sự thật, không bịa đặt thông tin (Accuracy)":          "Accuracy",
    "Trả lời đầy đủ, không bỏ sót ý quan trọng (Completeness)":          "Completeness",
    "Trả lời đúng trọng tâm câu hỏi, không lan man":                     "Relevance",
    "Cấu trúc trình bày rõ ràng (có khoảng trắng, gạch đầu dòng, in đậm...)": "Format",
    "Văn phong mạch lạc, tự nhiên, dễ hiểu (Naturalness)":                "Naturalness",
    "Ý kiến khác":                                                       "Ý kiến khác",
}

_PS_ERROR_OPTIONS = [
    "Cung cấp thông tin sai lệch, bịa đặt (Hallucination)",
    "Câu trả lời đúng nhưng văn phong lủng củng, không giống người Việt viết",
    "Câu trả lời quá ngắn, cụt lủn, thiếu thông tin cần thiết",
    "Câu trả lời quá dài dòng nhưng lại không đúng trọng tâm",
    "Tôi không nhận thấy lỗi nào đáng kể",
    "Lỗi khác",
]
_PS_ERROR_SHORT = {
    "Cung cấp thông tin sai lệch, bịa đặt (Hallucination)":                   "Hallucination",
    "Câu trả lời đúng nhưng văn phong lủng củng, không giống người Việt viết": "Văn phong lủng củng",
    "Câu trả lời quá ngắn, cụt lủn, thiếu thông tin cần thiết":                "Quá ngắn / thiếu",
    "Câu trả lời quá dài dòng nhưng lại không đúng trọng tâm":                 "Dài dòng / lan man",
    "Tôi không nhận thấy lỗi nào đáng kể":                                     "Không có lỗi",
    "Lỗi khác":                                                                "Lỗi khác",
}

_PS_CHEO_LEVEL_ORDER = [
    "1 - Hoàn toàn không biết gì",
    "2 - Biết rất ít (chỉ nghe tên, không rõ chi tiết)",
    "3 - Ở mức cơ bản (biết một vài vở diễn/nhân vật nổi tiếng)",
    "4 - Có hiểu biết khá tốt",
    "5 - Rất am hiểu (nghiên cứu hoặc làm việc liên quan đến nghệ thuật truyền thống)",
]
_PS_AI_USAGE_ORDER = [
    "Không bao giờ",
    "Hiếm khi",
    "Thỉnh thoảng (vài lần/tháng)",
    "Thường xuyên (vài lần/tuần)",
    "Hàng ngày",
]
_PS_TRUST_SHORT = {
    "Cột 1: GraphRAG":                                  "GraphRAG",
    "Cột 2: RAG":                                       "RAG",
    "Cột 3: LLM":                                       "LLM Only",
    "Không nhận thấy sự khác biệt rõ rệt giữa các cột": "Không khác biệt",
}
_PS_WAIT_ORDER = [
    "Rất nhanh, hoàn toàn thoải mái",
    "Hơi chậm nhưng ở mức chấp nhận được",
    "Quá chậm, làm giảm nghiêm trọng trải nghiệm của tôi",
]


def _cat_short(category: str) -> str:
    """'Dạng 1 — Tra cứu trực tiếp (Dễ)' → 'Dạng 1'."""
    if category.startswith("Câu hỏi tự do"):
        return "Câu hỏi tự do"
    return category.split("—")[0].strip() if "—" in category else category


def _user_key(name: str) -> str:
    """Canonical key cho mỗi người tham gia.

    Gộp các biến thể chỉ khác nhau ở dấu cách / dấu gạch / chữ hoa-thường:
        "24022981 - Nguyễn Phạm Sơn Hà"  ─┐
        "24022981-Nguyễn Phạm Sơn Hà"    ─┤→ "24022981 nguyễn phạm sơn hà"
        "24022981_Nguyễn Phạm Sơn Hà"    ─┘
    """
    if not name:
        return ""
    s = name.strip()
    # Tách mã SV (7-10 chữ số liền nhau) ra trước, để khỏi bị ảnh hưởng bởi vị trí.
    m = re.search(r"\d{7,10}", s)
    sid = m.group(0) if m else ""
    rest = re.sub(r"\d{7,10}", "", s)        # bỏ id khỏi phần còn lại
    rest = re.sub(r"[-_]+", " ", rest)        # các loại gạch → space
    rest = re.sub(r"\s+", " ", rest).strip().lower()
    return f"{sid} {rest}".strip()


def _pick_display_name(variants: list[str]) -> str:
    """Khi gộp, chọn biến thể có độ dài lớn nhất (nhiều thông tin nhất)."""
    return max((v for v in variants if v), key=len, default="")


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.an-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    border-radius: 16px; padding: 2rem 2rem 1.6rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
    flex-wrap: wrap;
}
.an-hero-left { flex: 1; min-width: 280px; }
.an-hero-title { font-size: 1.7rem; font-weight: 800; color: #fff; margin: 0 0 0.3rem; }
.an-hero-sub   { color: rgba(255,255,255,0.75); font-size: 0.95rem; margin: 0; }
.an-hero-meta  { color: rgba(255,255,255,0.55); font-size: 0.78rem; margin-top: 0.5rem; }

.an-kpi-grid { display: flex; gap: 12px; margin: 1rem 0 1.5rem; flex-wrap: wrap; }
.an-kpi {
    flex: 1; min-width: 160px; background: white;
    border: 1px solid #e5e9f0; border-radius: 12px;
    padding: 1rem 1.1rem;
}
.an-kpi-label { font-size: 0.78rem; color: #888; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.3px; }
.an-kpi-value { font-size: 2rem; font-weight: 800; color: #1a2a5e; line-height: 1.1;
                margin-top: 4px; }
.an-kpi-foot  { font-size: 0.78rem; color: #aaa; margin-top: 4px; }

.an-empty {
    background: #fafbfc; border: 1px dashed #d0d7e2; border-radius: 12px;
    padding: 2.5rem 1.5rem; text-align: center; color: #888;
}
</style>
"""


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_all() -> tuple[list[dict], list[dict]]:
    """Đọc toàn bộ JSON trong 2 folder. KHÔNG cache — đọc ~100 file JSON nhỏ
    chỉ tốn ~100ms, không đáng cache; bỏ cache để tránh stale data."""
    experiments: list[dict] = []
    preferences: list[dict] = []
    if _EXP_DIR.exists():
        for p in sorted(_EXP_DIR.glob("*.json")):
            try:
                experiments.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    if _PREF_DIR.exists():
        for p in sorted(_PREF_DIR.glob("*.json")):
            try:
                preferences.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return experiments, preferences


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _users_in(records: list[dict]) -> set[str]:
    """Trả về set canonical key — đã gộp các biến thể của cùng 1 người."""
    return {_user_key(r.get("user", "")) for r in records if r.get("user")}


def _preference_vote_table(prefs: list[dict]) -> pd.DataFrame:
    """1 row mỗi answer (user, question_id, category, best_system, ...)."""
    rows: list[dict] = []
    for rec in prefs:
        user = rec.get("user", "")
        for ans in rec.get("answers", []):
            if not ans.get("done"):
                continue
            rows.append({
                "user":        user,
                "question_id": ans.get("question_id", ""),
                "question":    ans.get("question", ""),
                "category":    ans.get("category", ""),
                "cat_short":   _cat_short(ans.get("category", "")),
                "best_system": ans.get("best_system", ""),
                "note":        ans.get("note", ""),
            })
    return pd.DataFrame(rows)


def _experiment_rating_table(exps: list[dict]) -> pd.DataFrame:
    """Long-form: 1 row mỗi (user, step, system, dimension)."""
    rows: list[dict] = []
    for rec in exps:
        user = rec.get("user", "")
        for step in rec.get("steps", []):
            cat = step.get("category", "")
            cat_s = _cat_short(cat)
            per_sys = step.get("rating", {}).get("per_system", {})
            for sk in _SYSTEMS:
                s = per_sys.get(sk, {})
                if not s:
                    continue
                for dim in _DIMENSIONS:
                    if dim in s and s[dim] is not None:
                        rows.append({
                            "user":      user,
                            "step":      step.get("step"),
                            "case_id":   step.get("case_id", ""),
                            "category":  cat,
                            "cat_short": cat_s,
                            "system":    sk,
                            "dimension": dim,
                            "rating":    s[dim],
                        })
    return pd.DataFrame(rows)


def _experiment_best_table(exps: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in exps:
        for step in rec.get("steps", []):
            best = step.get("rating", {}).get("best_system", "")
            if not best:
                continue
            rows.append({
                "user":        rec.get("user", ""),
                "step":        step.get("step"),
                "case_id":     step.get("case_id", ""),
                "category":    step.get("category", ""),
                "cat_short":   _cat_short(step.get("category", "")),
                "best_system": best,
            })
    return pd.DataFrame(rows)


def _experiment_latency_table(exps: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in exps:
        for step in rec.get("steps", []):
            for sk in _SYSTEMS:
                r = step.get("results", {}).get(sk, {})
                if not r or r.get("error"):
                    continue
                el = r.get("elapsed")
                if el is None:
                    continue
                rows.append({
                    "user":     rec.get("user", ""),
                    "step":     step.get("step"),
                    "system":   sk,
                    "elapsed":  float(el),
                })
    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _name_col(df: pd.DataFrame, src_col: str = "system") -> pd.Series:
    return df[src_col].map(_DISPLAY_NAMES).fillna(df[src_col])


def _system_color_map() -> dict[str, str]:
    return {_DISPLAY_NAMES[sk]: _DISPLAY_COLORS[sk] for sk in _SYSTEMS}




def _bar_total_votes(df: pd.DataFrame, *, title: str) -> go.Figure:
    """Build bar chart bằng go.Figure thuần — không qua plotly express,
    tránh hoàn toàn các quirk của px.bar khi split trace theo color."""
    if df is None or df.empty or "best_system" not in df.columns:
        counts = {sk: 0 for sk in _SYSTEMS}
    else:
        vc = df["best_system"].value_counts()
        counts = {sk: int(vc.get(sk, 0)) for sk in _SYSTEMS}

    fig = go.Figure()
    for sk in _SYSTEMS:
        v = counts[sk]
        fig.add_trace(go.Bar(
            x=[_DISPLAY_NAMES[sk]],
            y=[v],
            name=_DISPLAY_NAMES[sk],
            marker_color=_DISPLAY_COLORS[sk],
            text=[str(v)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Số phiếu: %{y}<extra></extra>",
        ))
    fig.update_layout(
        title=title, showlegend=False,
        xaxis_title=None, yaxis_title="Số phiếu",
        margin=dict(l=10, r=10, t=50, b=10), height=340,
    )
    return fig


def _grouped_by_category(df: pd.DataFrame, *, title: str) -> go.Figure:
    """Grouped column bar — mỗi mức độ có 3 cột (1/hệ thống)."""
    if df is None or df.empty:
        return go.Figure()

    cats_present = [c for c in _CAT_ORDER if c in df["cat_short"].unique()]
    fig = go.Figure()
    for sk in _SYSTEMS:
        ys = []
        texts = []
        for cat in cats_present:
            v = int(((df["cat_short"] == cat) & (df["best_system"] == sk)).sum())
            ys.append(v)
            texts.append(str(v))
        fig.add_trace(go.Bar(
            x=cats_present, y=ys,
            name=_DISPLAY_NAMES[sk],
            marker_color=_DISPLAY_COLORS[sk],
            text=texts, textposition="outside",
            hovertemplate="<b>%{x}</b> · " + _DISPLAY_NAMES[sk]
                          + "<br>Số phiếu: %{y}<extra></extra>",
        ))
    fig.update_layout(
        title=title, barmode="group",
        xaxis_title=None, yaxis_title="Số phiếu",
        legend_title_text="Hệ thống",
        margin=dict(l=10, r=10, t=50, b=10), height=380,
    )
    return fig


def _grouped_rating_bar(df: pd.DataFrame) -> go.Figure:
    """Điểm TB theo (system × dimension). Dựng go.Bar thủ công cho chắc."""
    if df is None or df.empty:
        return go.Figure()

    means = (df.assign(rating=df["rating"].astype(float))
               .groupby(["system", "dimension"])["rating"].mean())

    dim_labels = [_DIMENSION_LABELS[d] for d in _DIMENSIONS]
    fig = go.Figure()
    for sk in _SYSTEMS:
        ys = [float(means.get((sk, d), 0)) for d in _DIMENSIONS]
        texts = [f"{y:.2f}" for y in ys]
        fig.add_trace(go.Bar(
            x=dim_labels, y=ys,
            name=_DISPLAY_NAMES[sk],
            marker_color=_DISPLAY_COLORS[sk],
            text=texts, textposition="outside",
            hovertemplate="<b>%{x}</b> · " + _DISPLAY_NAMES[sk]
                          + "<br>Điểm TB: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title="Điểm trung bình theo từng chỉ số", barmode="group",
        xaxis_title=None, yaxis_title="Điểm trung bình (1-5)",
        yaxis=dict(range=[0, 5.4]),
        legend_title_text="Hệ thống",
        margin=dict(l=10, r=10, t=50, b=10), height=380,
    )
    return fig


def _radar_overall(df: pd.DataFrame) -> go.Figure:
    """Radar: trục = 3 chỉ số, mỗi hệ thống là 1 polygon."""
    if df.empty:
        return go.Figure()
    g = df.groupby(["system", "dimension"])["rating"].mean().unstack(fill_value=0)
    fig = go.Figure()
    for sk in _SYSTEMS:
        if sk not in g.index:
            continue
        vals = [g.loc[sk].get(d, 0) for d in _DIMENSIONS]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[_DIMENSION_LABELS[d] for d in _DIMENSIONS] + [_DIMENSION_LABELS[_DIMENSIONS[0]]],
            fill="toself",
            name=_DISPLAY_NAMES[sk],
            line=dict(color=_DISPLAY_COLORS[sk], width=2),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 5], visible=True)),
        showlegend=True, title="Tổng quan rating 3 chiều",
        margin=dict(l=20, r=20, t=50, b=20), height=380,
    )
    return fig


def _latency_box(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    df2 = df.copy()
    df2["name"] = _name_col(df2, "system")
    fig = px.box(
        df2, x="name", y="elapsed",
        color="name", color_discrete_map=_system_color_map(),
        title="Phân phối thời gian phản hồi (giây)",
        category_orders={"name": list(_DISPLAY_NAMES.values())},
        points="suspectedoutliers",
    )
    fig.update_layout(
        showlegend=False, xaxis_title=None, yaxis_title="Giây",
        margin=dict(l=10, r=10, t=50, b=10), height=380,
    )
    return fig


# ── Refresh action (gọi fetcher) ──────────────────────────────────────────────

def _run_fetch() -> tuple[bool, str]:
    """Chạy scripts.fetch_results.main(); trả (ok, output)."""
    import contextlib
    from scripts.fetch_results import main as fetch_main

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = fetch_main([])
        return rc == 0, buf.getvalue()
    except Exception as exc:  # pragma: no cover — surface any IMAP/auth error
        return False, f"{buf.getvalue()}\n{type(exc).__name__}: {exc}"


# ── Top hero + refresh ────────────────────────────────────────────────────────

def _render_hero(n_exp: int, n_pref: int, last_seen: datetime | None) -> None:
    last_str = (
        last_seen.strftime("%Y-%m-%d %H:%M") if last_seen else "—"
    )
    st.markdown(
        f'''
        <div class="an-hero">
          <div class="an-hero-left">
            <div class="an-hero-title">📈 Thống kê khảo sát</div>
            <div class="an-hero-sub">
              {n_exp} bài thử nghiệm · {n_pref} bài đánh giá ưu tiên
            </div>
            <div class="an-hero-meta">Cập nhật gần nhất: {last_str}</div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_refresh_bar() -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption(
            "Bấm **Cập nhật từ Gmail** để fetch các email kết quả mới và "
            "phân loại tự động vào 2 thư mục."
        )
    with c2:
        if st.button("🔄 Cập nhật từ Gmail", use_container_width=True, type="primary"):
            with st.spinner("Đang kết nối Gmail và tải email mới..."):
                ok, output = _run_fetch()
            if ok:
                st.cache_data.clear()
                st.success("Đã cập nhật xong. Tải lại số liệu...")
                with st.expander("Xem log fetch"):
                    st.code(output or "(không có output)", language="text")
                st.rerun()
            else:
                st.error("Fetch thất bại — xem log bên dưới.")
                st.code(output, language="text")
    with c3:
        if st.button("♻️ Tải lại từ disk", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


# ── KPI ───────────────────────────────────────────────────────────────────────

def _render_kpis(exps: list[dict], prefs: list[dict]) -> None:
    exp_users  = _users_in(exps)
    pref_users = _users_in(prefs)
    both       = exp_users & pref_users
    total      = exp_users | pref_users

    cards = [
        ("Tổng người tham gia", str(len(total)), "(unique)"),
        ("Bài thử nghiệm",      str(len(exps)),  f"{len(exp_users)} người"),
        ("Bài đánh giá ưu tiên", str(len(prefs)), f"{len(pref_users)} người"),
        ("Làm cả 2 phần",       str(len(both)),  f"{(len(both)/len(total)*100):.0f}% tổng" if total else "—"),
    ]
    html = '<div class="an-kpi-grid">'
    for label, value, foot in cards:
        html += (
            f'<div class="an-kpi">'
            f'  <div class="an-kpi-label">{label}</div>'
            f'  <div class="an-kpi-value">{value}</div>'
            f'  <div class="an-kpi-foot">{foot}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Tab 1 — Overview ──────────────────────────────────────────────────────────

def _render_overview_tab(exps: list[dict], prefs: list[dict]) -> None:
    if not exps and not prefs:
        st.markdown(
            '<div class="an-empty">Chưa có dữ liệu — bấm <strong>🔄 Cập nhật từ Gmail</strong> để tải về.</div>',
            unsafe_allow_html=True,
        )
        return

    pref_df = _preference_vote_table(prefs)
    exp_df  = _experiment_rating_table(exps)

    c1, c2 = st.columns(2)
    with c1:
        if not pref_df.empty:
            st.plotly_chart(
                _bar_total_votes(pref_df, title="Tổng vote — Đánh giá ưu tiên (21 câu)"),
                use_container_width=True,
            )
        else:
            st.info("Chưa có dữ liệu đánh giá ưu tiên.")
    with c2:
        if not exp_df.empty:
            st.plotly_chart(_radar_overall(exp_df), use_container_width=True)
        else:
            st.info("Chưa có dữ liệu thử nghiệm.")

    # Best-system distribution from experiment (4 câu)
    best_df = _experiment_best_table(exps)
    if not best_df.empty:
        st.plotly_chart(
            _bar_total_votes(best_df, title="Tổng vote — Thử nghiệm (4 câu/người)"),
            use_container_width=True,
        )


# ── Tab 2 — Preference detail ─────────────────────────────────────────────────

def _render_preference_tab(prefs: list[dict]) -> None:
    df = _preference_vote_table(prefs)
    if df.empty:
        st.markdown(
            '<div class="an-empty">Chưa có bài đánh giá ưu tiên nào.</div>',
            unsafe_allow_html=True,
        )
        return

    st.plotly_chart(
        _grouped_by_category(df, title="Vote theo mức độ câu hỏi"),
        use_container_width=True,
    )

    # ── Top consensus / contention ─────────────────────────────────────────
    by_q = (df.groupby(["question_id", "question", "best_system"])
              .size().unstack(fill_value=0))
    if not by_q.empty:
        for sk in _SYSTEMS:
            if sk not in by_q.columns:
                by_q[sk] = 0
        by_q["total"] = by_q[_SYSTEMS].sum(axis=1)
        by_q["winner"] = by_q[_SYSTEMS].idxmax(axis=1)
        by_q["winner_pct"] = (by_q[_SYSTEMS].max(axis=1) / by_q["total"] * 100).round(1)
        # Entropy (cao = tranh cãi)
        import numpy as np
        probs = by_q[_SYSTEMS].div(by_q["total"], axis=0).fillna(0)
        entropy = -(probs * np.where(probs > 0, np.log2(probs.where(probs > 0)), 0)).sum(axis=1)
        by_q["entropy"] = entropy.round(3)
        by_q = by_q.reset_index()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🏆 Câu hỏi đồng thuận cao nhất** (winner-pct cao)")
            top = by_q.nlargest(5, "winner_pct")[
                ["question", "winner", "winner_pct", "total"]
            ].copy()
            top["winner"] = top["winner"].map(_DISPLAY_NAMES)
            top.columns = ["Câu hỏi", "Hệ thống thắng", "% phiếu", "Tổng phiếu"]
            st.dataframe(top, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**⚔️ Câu hỏi tranh cãi nhất** (entropy cao)")
            top = by_q.nlargest(5, "entropy")[
                ["question", "winner", "winner_pct", "entropy", "total"]
            ].copy()
            top["winner"] = top["winner"].map(_DISPLAY_NAMES)
            top.columns = ["Câu hỏi", "Dẫn đầu", "% phiếu", "Entropy", "Tổng phiếu"]
            st.dataframe(top, use_container_width=True, hide_index=True)

    # ── Detail table — vote count per question ────────────────────────────
    with st.expander("📋 Chi tiết vote từng câu hỏi"):
        pivot = (df.pivot_table(
                    index=["question_id", "question", "cat_short"],
                    columns="best_system",
                    aggfunc="size",
                    fill_value=0,
                ).reset_index())
        for sk in _SYSTEMS:
            if sk not in pivot.columns:
                pivot[sk] = 0
            pivot.rename(columns={sk: _DISPLAY_NAMES[sk]}, inplace=True)
        pivot.rename(columns={
            "question_id": "Mã câu",
            "question":    "Câu hỏi",
            "cat_short":   "Mức độ",
        }, inplace=True)
        st.dataframe(pivot, use_container_width=True, hide_index=True)


# ── Tab 3 — Experiment detail ────────────────────────────────────────────────

def _render_experiment_tab(exps: list[dict]) -> None:
    rdf = _experiment_rating_table(exps)
    bdf = _experiment_best_table(exps)
    ldf = _experiment_latency_table(exps)

    if rdf.empty:
        st.markdown(
            '<div class="an-empty">Chưa có bài thử nghiệm nào.</div>',
            unsafe_allow_html=True,
        )
        return

    st.plotly_chart(_grouped_rating_bar(rdf), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _grouped_by_category(bdf, title="Best-system theo mức độ"),
            use_container_width=True,
        )
    with c2:
        if not ldf.empty:
            st.plotly_chart(_latency_box(ldf), use_container_width=True)
        else:
            st.info("Không có dữ liệu thời gian phản hồi.")

    # ── Per-category breakdown ────────────────────────────────────────────
    st.markdown("**Điểm trung bình theo mức độ câu hỏi**")
    means = (rdf.assign(rating=rdf["rating"].astype(float))
                .groupby(["cat_short", "system"])["rating"].mean())
    cats_present = [c for c in _CAT_ORDER if c in rdf["cat_short"].unique()]
    fig = go.Figure()
    for sk in _SYSTEMS:
        ys = [float(means.get((c, sk), 0)) for c in cats_present]
        texts = [f"{y:.2f}" for y in ys]
        fig.add_trace(go.Bar(
            x=cats_present, y=ys,
            name=_DISPLAY_NAMES[sk],
            marker_color=_DISPLAY_COLORS[sk],
            text=texts, textposition="outside",
            hovertemplate="<b>%{x}</b> · " + _DISPLAY_NAMES[sk]
                          + "<br>Điểm TB: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title=None, yaxis_title="Điểm TB (1-5)",
        yaxis=dict(range=[0, 5.4]),
        legend_title_text="Hệ thống",
        margin=dict(l=10, r=10, t=20, b=10), height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Câu hỏi tự do (FREE) ──────────────────────────────────────────────
    free_steps: list[dict] = []
    for rec in exps:
        for step in rec.get("steps", []):
            if step.get("case_id") == "FREE":
                free_steps.append({"user": rec.get("user", ""), "step": step})
    if free_steps:
        with st.expander(f"💬 Câu hỏi tự do ({len(free_steps)} câu)"):
            for it in free_steps:
                step = it["step"]
                rat = step.get("rating", {}).get("per_system", {})
                badges = "  ".join(
                    f'<span style="color:{_DISPLAY_COLORS[sk]};font-weight:600">'
                    f'{_DISPLAY_NAMES[sk]}: '
                    f'{rat.get(sk, {}).get("accuracy","-")}/'
                    f'{rat.get(sk, {}).get("completeness","-")}/'
                    f'{rat.get(sk, {}).get("naturalness","-")}'
                    f'</span>'
                    for sk in _SYSTEMS
                )
                best = step.get("rating", {}).get("best_system", "")
                best_lbl = _DISPLAY_NAMES.get(best, "?")
                st.markdown(
                    f"**{it['user']}** — *Tốt nhất: {best_lbl}*  \n"
                    f"❓ {step.get('question', '')}\n\n"
                    f"<small>{badges}</small>",
                    unsafe_allow_html=True,
                )
                st.divider()


# ── Tab 4 — Participants ─────────────────────────────────────────────────────

def _render_participants_tab(exps: list[dict], prefs: list[dict]) -> None:
    # Gộp theo canonical key — các biến thể "24022981 - X" / "24022981-X"
    # được nhập làm cùng một người. Chỉ lấy bản nộp mới nhất của mỗi loại.
    by_user: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "experiment": None, "preference": None, "names": [],
    })

    def _take_latest(slot: dict, kind: str, rec: dict) -> None:
        """Ghi đè rec hiện tại nếu rec mới có last_updated muộn hơn."""
        existing = slot[kind]
        new_ts = rec.get("last_updated", "")
        if existing is None or new_ts > existing.get("last_updated", ""):
            slot[kind] = rec

    for rec in exps:
        u = rec.get("user", "")
        if not u:
            continue
        slot = by_user[_user_key(u)]
        _take_latest(slot, "experiment", rec)
        slot["names"].append(u)
    for rec in prefs:
        u = rec.get("user", "")
        if not u:
            continue
        slot = by_user[_user_key(u)]
        _take_latest(slot, "preference", rec)
        slot["names"].append(u)

    # Pick display name cho mỗi canonical key
    display_of = {k: _pick_display_name(v["names"]) for k, v in by_user.items()}

    if not by_user:
        st.markdown(
            '<div class="an-empty">Chưa có người tham gia nào.</div>',
            unsafe_allow_html=True,
        )
        return

    rows = []
    for key in sorted(by_user.keys(), key=lambda k: display_of[k].lower()):
        slots = by_user[key]
        exp_rec  = slots["experiment"]
        pref_rec = slots["preference"]
        pref_completed = pref_rec.get("completed", 0) if pref_rec else 0
        pref_total     = pref_rec.get("total_questions", 21) if pref_rec else 21
        exp_when  = (exp_rec or {}).get("last_updated", "")
        pref_when = (pref_rec or {}).get("last_updated", "")
        rows.append({
            "Người tham gia":  display_of[key],
            "Thử nghiệm":      "✅ 4/4" if exp_rec else "—",
            "Đánh giá ưu tiên": f"✅ {pref_completed}/{pref_total}" if pref_rec else "—",
            "Nộp TN":          exp_when[:16].replace("T", " ") if exp_when else "",
            "Nộp ĐG":          pref_when[:16].replace("T", " ") if pref_when else "",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Drill-down ────────────────────────────────────────────────────────
    st.markdown("---")
    keys_sorted = sorted(by_user.keys(), key=lambda k: display_of[k].lower())
    pick_label = st.selectbox(
        "Xem chi tiết của:",
        options=keys_sorted,
        format_func=lambda k: display_of[k],
        key="an_user_pick",
    )
    if not pick_label:
        return
    slots = by_user[pick_label]

    exp_rec, pref_rec = slots["experiment"], slots["preference"]

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### 🧪 Thử nghiệm")
        if not exp_rec:
            st.caption("(Chưa nộp)")
        else:
            for step in exp_rec.get("steps", []):
                rat = step.get("rating", {})
                ps = rat.get("per_system", {})
                best = _DISPLAY_NAMES.get(rat.get("best_system", ""), "?")
                st.markdown(
                    f"**Bước {step.get('step')}** — `{step.get('case_id', '')}`"
                    f" · *Tốt nhất: {best}*  \n"
                    f"{step.get('question', '')}"
                )
                row = {
                    _DIMENSION_LABELS[d]: [
                        ps.get(sk, {}).get(d, "-") for sk in _SYSTEMS
                    ] for d in _DIMENSIONS
                }
                tbl = pd.DataFrame(
                    row, index=[_DISPLAY_NAMES[sk] for sk in _SYSTEMS]
                )
                st.table(tbl)
                if rat.get("note"):
                    st.caption(f"📝 {rat['note']}")

    with cols[1]:
        st.markdown("#### 📋 Đánh giá ưu tiên")
        if not pref_rec:
            st.caption("(Chưa nộp)")
        else:
            answers = pref_rec.get("answers", [])
            done_n = sum(1 for a in answers if a.get("done"))
            total = pref_rec.get("total_questions", 21)
            st.caption(f"Hoàn thành: {done_n}/{total}")
            votes = Counter(
                a.get("best_system", "") for a in answers if a.get("done")
            )
            tbl = pd.DataFrame({
                "Hệ thống": [_DISPLAY_NAMES[sk] for sk in _SYSTEMS],
                "Số phiếu": [votes.get(sk, 0) for sk in _SYSTEMS],
            })
            st.table(tbl)
            with st.expander("Xem từng câu"):
                qa = [{
                    "Mã":    a.get("question_id", ""),
                    "Câu hỏi": a.get("question", "")[:60] + (
                        "..." if len(a.get("question", "")) > 60 else ""
                    ),
                    "Vote":  _DISPLAY_NAMES.get(a.get("best_system", ""), "—"),
                    "Ghi chú": a.get("note", ""),
                } for a in answers if a.get("done")]
                st.dataframe(pd.DataFrame(qa), use_container_width=True, hide_index=True)


# ── Post-survey: load / parse / plot ─────────────────────────────────────────

def _load_post_survey() -> pd.DataFrame | None:
    p = _find_post_survey_csv()
    if p is None:
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_multi_choice(series: pd.Series, options: list[str]) -> dict[str, int]:
    """Đếm số response chứa từng option canonical.
    Match theo substring vì Google Forms join multi-select bằng ', ' mà chính
    options cũng chứa ', ' nên không split được."""
    counts = {opt: 0 for opt in options}
    for s in series.dropna():
        text = str(s)
        for opt in options:
            if opt in text:
                counts[opt] += 1
    return counts


def _ordered_bar(counts: dict[str, int], *, title: str,
                 color: str = "#3498db", horizontal: bool = False) -> go.Figure:
    """Bar chart đơn — counts theo thứ tự đã có sẵn trong dict."""
    labels = list(counts.keys())
    values = list(counts.values())
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(
            y=labels, x=values, orientation="h",
            marker_color=color,
            text=[str(v) for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Số phiếu: %{x}<extra></extra>",
        ))
        fig.update_layout(
            title=title, showlegend=False,
            xaxis_title="Số phiếu", yaxis_title=None,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=50, b=10), height=420,
        )
    else:
        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=color,
            text=[str(v) for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Số phiếu: %{y}<extra></extra>",
        ))
        fig.update_layout(
            title=title, showlegend=False,
            xaxis_title=None, yaxis_title="Số phiếu",
            margin=dict(l=10, r=10, t=50, b=10), height=380,
        )
    return fig


def _colored_bar(counts: dict[str, int], colors: dict[str, str], *, title: str,
                 yaxis_title: str = "Số phiếu") -> go.Figure:
    """Bar chart với màu riêng cho từng cột."""
    fig = go.Figure()
    for lbl, val in counts.items():
        fig.add_trace(go.Bar(
            x=[lbl], y=[val],
            marker_color=colors.get(lbl, "#888"),
            name=lbl,
            text=[str(val)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + yaxis_title + ": %{y}<extra></extra>",
        ))
    fig.update_layout(
        title=title, showlegend=False,
        xaxis_title=None, yaxis_title=yaxis_title,
        margin=dict(l=10, r=10, t=50, b=10), height=380,
    )
    return fig


def _render_post_survey_tab() -> None:
    df = _load_post_survey()
    if df is None or df.empty:
        st.markdown(
            '<div class="an-empty">Chưa tìm thấy file CSV khảo sát hậu nghiệm.</div>',
            unsafe_allow_html=True,
        )
        return

    st.caption(f"📊 {len(df)} phản hồi · file: `{_find_post_survey_csv().name}`")

    # ── 1. Demographics ─────────────────────────────────────────────────────
    st.markdown("### 1. Hồ sơ người tham gia")
    c1, c2 = st.columns(2)
    with c1:
        vc = df[_PS_COL_CHEO_KNOWLEDGE].value_counts()
        counts = {lvl: int(vc.get(lvl, 0)) for lvl in _PS_CHEO_LEVEL_ORDER if vc.get(lvl, 0) > 0}
        # Rút gọn label cho dễ đọc
        short = {k: k.split(" - ")[0] + " - " + k.split(" - ")[1].split(" (")[0]
                 for k in counts.keys()}
        counts_short = {short[k]: v for k, v in counts.items()}
        st.plotly_chart(
            _ordered_bar(counts_short, title="Mức độ hiểu biết về nghệ thuật Chèo",
                         color="#9b59b6"),
            use_container_width=True,
        )
    with c2:
        vc = df[_PS_COL_AI_USAGE].value_counts()
        counts = {lvl: int(vc.get(lvl, 0)) for lvl in _PS_AI_USAGE_ORDER if vc.get(lvl, 0) > 0}
        st.plotly_chart(
            _ordered_bar(counts, title="Tần suất sử dụng công cụ AI",
                         color="#16a085"),
            use_container_width=True,
        )

    # ── 2. Hệ thống đáng tin cậy nhất ───────────────────────────────────────
    st.markdown("### 2. Hệ thống đem lại cảm giác đáng tin cậy nhất")
    vc = df[_PS_COL_TRUSTED_SYS].value_counts()
    counts = {_PS_TRUST_SHORT.get(k, k): int(v) for k, v in vc.items()}
    # Sắp xếp theo thứ tự cố định
    order = ["GraphRAG", "RAG", "LLM Only", "Không khác biệt"]
    counts_ordered = {k: counts.get(k, 0) for k in order if k in counts}
    color_map = {
        "GraphRAG":        "#2ecc71",
        "RAG":             "#3498db",
        "LLM Only":        "#e67e22",
        "Không khác biệt": "#95a5a6",
    }
    st.plotly_chart(
        _colored_bar(counts_ordered, color_map,
                     title="Số phiếu bình chọn hệ thống đáng tin cậy nhất"),
        use_container_width=True,
    )

    # ── 3. Tiêu chí + Lỗi (multi-select) ────────────────────────────────────
    st.markdown("### 3. Tiêu chí chọn best & các lỗi gặp phải")
    c1, c2 = st.columns(2)
    with c1:
        crit_counts_full = _parse_multi_choice(df[_PS_COL_CRITERIA], _PS_CRITERIA_OPTIONS)
        crit_counts = {_PS_CRITERIA_SHORT[k]: v for k, v in crit_counts_full.items() if v > 0}
        crit_counts = dict(sorted(crit_counts.items(), key=lambda x: -x[1]))
        st.plotly_chart(
            _ordered_bar(crit_counts, title="Tiêu chí ưu tiên khi chọn 'Hệ thống tốt nhất'",
                         color="#3498db", horizontal=True),
            use_container_width=True,
        )
    with c2:
        err_counts_full = _parse_multi_choice(df[_PS_COL_ERRORS], _PS_ERROR_OPTIONS)
        err_counts = {_PS_ERROR_SHORT[k]: v for k, v in err_counts_full.items() if v > 0}
        err_counts = dict(sorted(err_counts.items(), key=lambda x: -x[1]))
        st.plotly_chart(
            _ordered_bar(err_counts, title="Các lỗi gặp phải trong câu trả lời",
                         color="#e74c3c", horizontal=True),
            use_container_width=True,
        )

    # ── 4. UX (UI rating + thời gian chờ) ───────────────────────────────────
    st.markdown("### 4. Trải nghiệm người dùng (UX)")
    c1, c2 = st.columns(2)
    with c1:
        vc = df[_PS_COL_UI_RATING].value_counts().sort_index()
        counts = {f"{int(k)}/5": int(v) for k, v in vc.items()}
        color_map = {f"{i}/5": c for i, c in zip(
            range(1, 6),
            ["#e74c3c", "#e67e22", "#f39c12", "#27ae60", "#16a085"]
        )}
        st.plotly_chart(
            _colored_bar(counts, color_map,
                         title="Đánh giá mức độ thân thiện của giao diện"),
            use_container_width=True,
        )
    with c2:
        vc = df[_PS_COL_WAIT_TIME].value_counts()
        counts = {k: int(vc.get(k, 0)) for k in _PS_WAIT_ORDER if vc.get(k, 0) > 0}
        # Rút gọn label
        short = {
            "Rất nhanh, hoàn toàn thoải mái":                       "Rất nhanh",
            "Hơi chậm nhưng ở mức chấp nhận được":                  "Hơi chậm (chấp nhận được)",
            "Quá chậm, làm giảm nghiêm trọng trải nghiệm của tôi":  "Quá chậm",
        }
        counts_short = {short.get(k, k): v for k, v in counts.items()}
        color_map = {
            "Rất nhanh":                  "#27ae60",
            "Hơi chậm (chấp nhận được)":  "#f39c12",
            "Quá chậm":                   "#e74c3c",
        }
        st.plotly_chart(
            _colored_bar(counts_short, color_map,
                         title="Cảm nhận về thời gian chờ phản hồi"),
            use_container_width=True,
        )

    # ── 5. Ý kiến mở (text) ─────────────────────────────────────────────────
    st.markdown("### 5. Ý kiến mở")
    impressions = df[_PS_COL_IMPRESSION].dropna().astype(str).str.strip()
    impressions = [t for t in impressions if t]
    feedbacks = df[_PS_COL_FEEDBACK].dropna().astype(str).str.strip()
    feedbacks = [t for t in feedbacks if t]

    with st.expander(f"💬 Ấn tượng / Thất vọng ({len(impressions)} ý kiến)"):
        for t in impressions:
            st.markdown(f"- {t}")
    with st.expander(f"📝 Góp ý hoàn thiện ({len(feedbacks)} ý kiến)"):
        for t in feedbacks:
            st.markdown(f"- {t}")


# ── Main render ──────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    exps, prefs = _load_all()

    last_seen = None
    for rec in exps + prefs:
        ts = rec.get("last_updated")
        if not ts:
            continue
        try:
            d = datetime.fromisoformat(ts)
            if last_seen is None or d > last_seen:
                last_seen = d
        except ValueError:
            continue

    _render_hero(len(exps), len(prefs), last_seen)
    _render_refresh_bar()
    _render_kpis(exps, prefs)

    tabs = st.tabs([
        "📊 Tổng quan",
        "📋 Đánh giá ưu tiên",
        "🧪 Thử nghiệm",
        "👥 Người tham gia",
        "🗳️ Khảo sát hậu nghiệm",
    ])
    with tabs[0]:
        _render_overview_tab(exps, prefs)
    with tabs[1]:
        _render_preference_tab(prefs)
    with tabs[2]:
        _render_experiment_tab(exps)
    with tabs[3]:
        _render_participants_tab(exps, prefs)
    with tabs[4]:
        _render_post_survey_tab()
