"""
👋 Intro page — trang chào dành cho người tham gia thực nghiệm (GUEST_MODE).

Giới thiệu tổng quan cơ sở dữ liệu Chèo + hướng dẫn 2 phần thực nghiệm.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_ENTITIES_FILE = Path(__file__).resolve().parents[1] / "data" / "cheo_entities.txt"


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_entities() -> dict[str, list[str]]:
    """Parse cheo_entities.txt into {category: [names...]}."""
    result: dict[str, list[str]] = {
        "plays": [], "scenes": [], "characters": [], "actors": [],
    }
    if not _ENTITIES_FILE.exists():
        return result
    key_map = {
        "VỞ CHÈO": "plays",
        "TRÍCH ĐOẠN": "scenes",
        "NHÂN VẬT": "characters",
        "DIỄN VIÊN": "actors",
    }
    current: str | None = None
    with open(_ENTITIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            header = line.rstrip(":")
            if header in key_map:
                current = key_map[header]
                continue
            if current:
                result[current].extend(
                    n.strip() for n in line.split(",") if n.strip()
                )
    return result


def _nav_to(page: str) -> None:
    st.session_state["_nav_pending"] = page
    st.rerun()


# ── Render helpers ────────────────────────────────────────────────────────────

def _render_tags(items: list[str], bg: str, fg: str, border: str) -> None:
    """Render a list of names as compact pill-style tags."""
    html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 12px'>"
    for name in items:
        html += (
            f"<span style='background:{bg};color:{fg};border:1px solid {border};"
            f"border-radius:999px;padding:3px 12px;font-size:0.82rem;"
            f"font-weight:500;white-space:nowrap'>{name}</span>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    entities = _parse_entities()

    # ── 1. Hero ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #8b0000 0%, #c0392b 45%, #e67e22 100%);
            padding: 2.8rem 2rem 2.4rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.20);
        ">
            <div style="font-size: 3.6rem; margin-bottom: 0.3rem; line-height:1">🎭</div>
            <h1 style="
                color: #ffffff; font-size: 2.2rem; font-weight: 800;
                margin: 0 0 0.5rem; letter-spacing: -0.5px;
                text-shadow: 0 2px 8px rgba(0,0,0,0.3);
            ">Chào mừng đến với Thực nghiệm GraphRAG Chèo</h1>
            <p style="color: #fff3e0; font-size: 1.05rem; margin: 0 0 0.3rem; font-weight: 400;">
                Hệ thống Hỏi đáp về Nghệ thuật Chèo Việt Nam — Nghiên cứu so sánh 3 phương pháp AI
            </p>
            <p style="color: #ffd8b0; font-size: 0.9rem; margin: 0;">
                Cảm ơn bạn đã tham gia! Trang này giới thiệu dữ liệu và các bước thực nghiệm.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Về nghệ thuật Chèo ─────────────────────────────────────────────────
    st.markdown("### 🎭 Về nghệ thuật Chèo")
    st.markdown(
        """
        **Chèo** là một trong những loại hình **sân khấu dân gian truyền thống** lâu đời nhất
        của người Việt ở đồng bằng Bắc Bộ. Chèo kết hợp hát, múa, nhạc và kịch với các
        nhân vật điển hình như *Đào*, *Kép*, *Lão*, *Mụ*, *Hề* — phản ánh đời sống,
        tín ngưỡng và nhân sinh quan của người dân qua nhiều thế kỷ.

        Nghiên cứu này xây dựng **Knowledge Graph** (đồ thị tri thức) về các vở chèo cổ,
        nhằm đánh giá xem hệ thống **GraphRAG** (RAG kết hợp đồ thị tri thức)
        có trả lời câu hỏi về Chèo tốt hơn so với **RAG truyền thống** và **LLM thuần** hay không.
        """
    )

    st.divider()

    # ── 3. Thống kê cơ sở dữ liệu ─────────────────────────────────────────────
    st.markdown("### 📊 Cơ sở dữ liệu Chèo trong hệ thống")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📖 Vở chèo",    len(entities["plays"]),      help="Số vở chèo cổ đã được số hoá vào Knowledge Graph")
    c2.metric("🎬 Trích đoạn", len(entities["scenes"]),     help="Số trích đoạn (cảnh) được biểu diễn và ghi hình")
    c3.metric("🎭 Nhân vật",   len(entities["characters"]), help="Tổng số nhân vật xuất hiện trong các vở")
    c4.metric("🎤 Diễn viên",  len(entities["actors"]),     help="Số nghệ sĩ Chèo được ghi nhận trong cơ sở dữ liệu")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3a. Danh sách vở chèo ─────────────────────────────────────────────────
    if entities["plays"]:
        st.markdown("#### 📖 Danh sách vở chèo cổ")
        _render_tags(
            entities["plays"],
            bg="#fff5e6", fg="#8b4513", border="#f4a460",
        )

    # ── 3b. Danh sách trích đoạn ──────────────────────────────────────────────
    if entities["scenes"]:
        st.markdown("#### 🎬 Danh sách trích đoạn")
        _render_tags(
            entities["scenes"],
            bg="#fdf5ff", fg="#6a1b9a", border="#c890e0",
        )

    # ── 3c. Nhân vật (expander — danh sách dài) ──────────────────────────────
    if entities["characters"]:
        with st.expander(f"🎭 Xem danh sách {len(entities['characters'])} nhân vật"):
            _render_tags(
                entities["characters"],
                bg="#fef3c7", fg="#92400e", border="#f59e0b",
            )

    # ── 3d. Diễn viên (expander — danh sách dài) ─────────────────────────────
    if entities["actors"]:
        with st.expander(f"🎤 Xem danh sách {len(entities['actors'])} diễn viên"):
            _render_tags(
                entities["actors"],
                bg="#e8f0fe", fg="#1565c0", border="#90b8f0",
            )

    st.markdown(
        f"""
        <div style='background:#f8fafc;border-left:4px solid #2c5364;
             padding:12px 16px;border-radius:6px;margin-top:8px;font-size:0.9rem;color:#334155'>
        💡 <b>Knowledge Graph</b> liên kết 4 loại thực thể trên bằng các quan hệ như
        <i>"thuộc vở"</i>, <i>"đóng vai"</i>, <i>"xuất hiện trong trích đoạn"</i>…
        cho phép trả lời những câu hỏi phức tạp như
        <i>"diễn viên nào đóng nhiều vai nhất?"</i> hay
        <i>"nhân vật X xuất hiện trong những vở nào?"</i>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 4. Các bước thực nghiệm ───────────────────────────────────────────────
    st.markdown("### 🧭 Bạn sẽ tham gia 2 phần thực nghiệm")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div style="
                background:#fffbeb;border:1px solid #f59e0b;
                border-radius:12px;padding:1.2rem 1.2rem 1rem;min-height:200px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
            ">
              <div style="font-size:1.8rem;margin-bottom:0.4rem">🧪</div>
              <div style="font-weight:700;font-size:1.05rem;color:#78350f;margin-bottom:0.5rem">
                Phần 1 — Thử nghiệm & Chấm điểm
              </div>
              <div style="font-size:0.88rem;color:#4a5568;line-height:1.55">
                Bạn sẽ được hỏi <b>21 câu</b> về Chèo (7 câu dễ, 7 trung bình, 7 khó)
                + câu hỏi tự do. Với mỗi câu, <b>3 hệ thống AI</b> sẽ trả lời song song —
                bạn chấm điểm từng câu trả lời theo các tiêu chí:
                <i>chính xác, đầy đủ, dễ hiểu, hữu ích</i>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div style="
                background:#f5fff8;border:1px solid #90d8a8;
                border-radius:12px;padding:1.2rem 1.2rem 1rem;min-height:200px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
            ">
              <div style="font-size:1.8rem;margin-bottom:0.4rem">📋</div>
              <div style="font-weight:700;font-size:1.05rem;color:#14532d;margin-bottom:0.5rem">
                Phần 2 — Đánh giá ưu tiên (A/B/C)
              </div>
              <div style="font-size:0.88rem;color:#4a5568;line-height:1.55">
                Với cùng các câu hỏi trên, bạn sẽ thấy <b>3 câu trả lời ẩn tên</b>
                (gọi là A, B, C) và <b>chọn câu trả lời bạn ưa thích nhất</b>.
                Phần này giúp so sánh trực tiếp các hệ thống mà không bị ảnh hưởng
                bởi định kiến về tên hệ thống.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "⏱️ **Thời gian dự kiến:** 20-30 phút cho cả 2 phần. "
        "Bạn có thể dừng bất cứ lúc nào — tiến độ được lưu tự động theo tên bạn nhập. "
        "Kết quả sẽ được gửi về nhóm nghiên cứu qua email để phân tích."
    )

    # ── 5. Call to action ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    cta_col1, cta_col2, _ = st.columns([1.3, 1.3, 1])
    with cta_col1:
        if st.button(
            "🧪 Bắt đầu Phần 1 — Thử nghiệm",
            use_container_width=True,
            type="primary",
        ):
            _nav_to("🧪 Thử nghiệm")
    with cta_col2:
        if st.button(
            "📋 Đi thẳng đến Phần 2 — Đánh giá ưu tiên",
            use_container_width=True,
        ):
            _nav_to("📋 Đánh giá ưu tiên")

    st.markdown(
        """
        <div style='text-align:center;margin-top:2rem;padding-top:1rem;
             border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.82rem'>
        🎓 Nghiên cứu khoá luận tốt nghiệp · GraphRAG cho Nghệ thuật Chèo Việt Nam
        </div>
        """,
        unsafe_allow_html=True,
    )
