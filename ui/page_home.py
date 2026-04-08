"""
🏠 Home page — Giới thiệu hệ thống GraphRAGv2.

Trang mở đầu gồm 5 section:
  1. Hero banner + nút điều hướng nhanh
  2. Thống kê Knowledge Graph (từ file hoặc Neo4j)
  3. Feature cards cho 4 trang chức năng
  4. Câu hỏi mẫu theo danh mục (click → prefill GraphRAG)
  5. Hướng dẫn bắt đầu nhanh
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_ENTITIES_FILE = Path(__file__).resolve().parents[1] / "data" / "cheo_entities.txt"

# ── Sample questions ──────────────────────────────────────────────────────────

_SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "🎭 Nhân vật": [
        "Nhân vật Thị Mầu có đặc điểm gì trong nghệ thuật Chèo?",
        "Thị Kính bị oan trong vở chèo nào và như thế nào?",
        "Châu Long nuôi bạn thay chồng trong hoàn cảnh gì?",
        "Nhân vật Súy Vân là ai và tại sao phải giả dại?",
    ],
    "📖 Vở kịch": [
        "Vở chèo Quan Âm Thị Kính có những nhân vật chính nào?",
        "Vở Lưu Bình - Dương Lễ kể về tình bạn như thế nào?",
        "Liệt kê tất cả các vở chèo trong hệ thống.",
        "Vở Kim Nham có nội dung về chủ đề gì?",
    ],
    "🎤 Diễn viên": [
        "Diễn viên Thanh Ngoan đã thủ vai những nhân vật gì?",
        "Ai là diễn viên đóng vai Thị Mầu trong các phiên bản?",
        "Diễn viên Hồng Thắm tham gia các vở nào?",
        "Liệt kê các diễn viên đã thủ vai nhân vật Thị Kính.",
    ],
    "🎬 Trích đoạn": [
        "Trích đoạn Thị Mầu lên chùa kể về sự kiện gì?",
        "Súy Vân giả dại trong trích đoạn nào và vì sao?",
        "Trích đoạn Tuần Ty - Đào Huế có những nhân vật nào?",
        "Mô tả nội dung trích đoạn Vu quy.",
    ],
    "🔀 Tổng hợp": [
        "Liệt kê tất cả nhân vật và diễn viên xuất hiện trong vở Quan Âm Thị Kính.",
        "Trích đoạn Thị Mầu lên chùa có bao nhiêu phiên bản và ai diễn?",
        "Diễn viên nào đã thủ vai trong nhiều trích đoạn nhất?",
        "Nhân vật nào xuất hiện trong nhiều trích đoạn khác nhau?",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_entity_counts() -> dict[str, int]:
    """Count entities from the static fallback file."""
    counts = {"plays": 0, "scenes": 0, "characters": 0, "actors": 0}
    if not _ENTITIES_FILE.exists():
        return counts
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
                names = [n.strip() for n in line.split(",") if n.strip()]
                counts[current] += len(names)
    return counts


@st.cache_data(ttl=300, show_spinner=False)
def _get_neo4j_stats() -> dict | None:
    """Try to fetch schema stats from Neo4j. Returns None if unavailable."""
    try:
        from src.graph_loader.neo4j_client import Neo4jClient
        client = Neo4jClient()
        client.ping()
        schema = client.get_schema_summary()
        client.close()
        return schema
    except Exception:
        return None


def _nav_to(page: str, prefill_key: str = "", prefill_val: str = "") -> None:
    """Navigate to another page, optionally setting a prefill value."""
    st.session_state["_nav_page"] = page
    if prefill_key and prefill_val:
        st.session_state[prefill_key] = prefill_val
    st.rerun()


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:

    # ── Section 1: Hero ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
            padding: 3rem 2rem 2.5rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 0.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        ">
            <div style="font-size: 3.8rem; margin-bottom: 0.4rem; line-height:1">🎭</div>
            <h1 style="
                color: #ffffff;
                font-size: 2.4rem;
                font-weight: 800;
                margin: 0 0 0.5rem;
                letter-spacing: -0.5px;
                text-shadow: 0 2px 8px rgba(0,0,0,0.4);
            ">GraphRAG Chèo</h1>
            <p style="
                color: #b8d4e8;
                font-size: 1.1rem;
                margin: 0 0 0.3rem;
                font-weight: 400;
            ">Hệ thống Hỏi đáp thông minh về Nghệ thuật Chèo Việt Nam</p>
            <p style="
                color: #7a9ab5;
                font-size: 0.9rem;
                margin: 0;
            ">Knowledge Graph + Large Language Model · Câu trả lời chính xác, minh bạch</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        if st.button("⚖️ So sánh 3 hệ thống", use_container_width=True, type="primary"):
            _nav_to("⚖️ So sánh")
    with col1:
        if st.button("🔍 Dùng GraphRAG", use_container_width=True):
            _nav_to("🔍 GraphRAG")
    with col2:
        if st.button("📚 Dùng RAG truyền thống", use_container_width=True):
            _nav_to("📚 RAG")
    with col3:
        if st.button("💬 Chat với AI", use_container_width=True):
            _nav_to("💬 Chat")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Knowledge Graph Stats ──────────────────────────────────────
    st.markdown("### 📊 Dữ liệu Knowledge Graph")

    counts = _parse_entity_counts()
    neo4j_stats = _get_neo4j_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📖 Vở chèo",    counts["plays"],      help="Tổng số vở chèo trong ontology")
    c2.metric("🎬 Trích đoạn", counts["scenes"],     help="Số trích đoạn (cảnh) đã được đánh chỉ mục")
    c3.metric("🎭 Nhân vật",   counts["characters"], help="Tổng số nhân vật trong các vở")
    c4.metric("🎤 Diễn viên",  counts["actors"],     help="Số diễn viên được ghi nhận")

    if neo4j_stats:
        st.success(
            f"🔗 **Neo4j đang kết nối** — "
            f"**{neo4j_stats['total_nodes']:,}** nodes · "
            f"**{neo4j_stats['total_relationships']:,}** relationships · "
            f"Kiểu node: {', '.join(f'`{lb}`' for lb in sorted(neo4j_stats['node_labels']))}"
        )
    else:
        st.info(
            "🔗 **Knowledge Graph:** OWL Ontology Chèo 2025 · 7 kiểu node · 7 kiểu quan hệ · "
            "_Vào trang **🔗 Neo4j** để kết nối và xem thống kê đầy đủ._"
        )

    st.divider()

    # ── Section 3: Feature Cards ───────────────────────────────────────────────
    st.markdown("### 🧩 Các trang chức năng")

    _CARDS = [
        {
            "icon": "⚖️",
            "title": "So sánh",
            "badge": "⭐ Thực nghiệm",
            "badge_fg": "#b45309",
            "badge_bg": "#fef3c7",
            "border": "#f59e0b",
            "bg": "#fffbeb",
            "desc": (
                "Hỏi <b>1 câu</b>, cả 3 hệ thống <b>trả lời song song</b>. "
                "Kết quả hiện cạnh nhau để so sánh trực quan "
                "GraphRAG vs RAG vs LLM."
            ),
            "page": "⚖️ So sánh",
        },
        {
            "icon": "🔍",
            "title": "GraphRAG",
            "badge": "⭐ Chính",
            "badge_fg": "#c05000",
            "badge_bg": "#fff0e0",
            "border": "#f5c080",
            "bg": "#fffbf5",
            "desc": (
                "Truy xuất thông tin từ <b>Knowledge Graph</b> Chèo, "
                "sau đó dùng LLM tổng hợp câu trả lời. "
                "Xem được đồ thị tri thức nào được sử dụng qua tab <i>🕸️ Đồ thị</i>."
            ),
            "page": "🔍 GraphRAG",
        },
        {
            "icon": "📚",
            "title": "RAG truyền thống",
            "badge": "Baseline",
            "badge_fg": "#1565c0",
            "badge_bg": "#e8f0fe",
            "border": "#90b8f0",
            "bg": "#f5f9ff",
            "desc": (
                "Tìm kiếm <b>vector similarity</b> trong kho văn bản, "
                "dùng LLM trả lời từ các đoạn liên quan nhất. "
                "So sánh với GraphRAG để thấy điểm mạnh của graph."
            ),
            "page": "📚 RAG",
        },
        {
            "icon": "💬",
            "title": "Chat với AI",
            "badge": "Baseline",
            "badge_fg": "#1b6b3a",
            "badge_bg": "#e8f5e9",
            "border": "#90d8a8",
            "bg": "#f5fff8",
            "desc": (
                "<b>LLM thuần</b> — không có dữ liệu Chèo bổ sung. "
                "Dùng để so sánh giữa 3 hệ thống: LLM thuần, "
                "RAG vector và GraphRAG."
            ),
            "page": "💬 Chat",
        },
        {
            "icon": "📊",
            "title": "Benchmark",
            "badge": "Nghiên cứu",
            "badge_fg": "#6a1b9a",
            "badge_bg": "#f3e5f5",
            "border": "#c890e0",
            "bg": "#fdf5ff",
            "desc": (
                "Chạy <b>đánh giá định lượng</b> trên bộ CheoBench 100 câu. "
                "So sánh GraphRAG vs RAG theo IR, NLG, Exact metrics. "
                "Xuất kết quả JSON/CSV."
            ),
            "page": "📊 Benchmark",
        },
    ]

    cols = st.columns(len(_CARDS))

    for col, card in zip(cols, _CARDS):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: {card['bg']};
                    border: 1px solid {card['border']};
                    border-radius: 12px;
                    padding: 1.2rem 1rem 0.8rem;
                    min-height: 220px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                ">
                    <div style="font-size:1.9rem;margin-bottom:0.3rem">{card['icon']}</div>
                    <div style="display:flex;align-items:center;gap:6px;margin-bottom:0.6rem;flex-wrap:wrap">
                        <strong style="font-size:0.97rem">{card['title']}</strong>
                        <span style="
                            background:{card['badge_bg']};
                            color:{card['badge_fg']};
                            border:1px solid {card['badge_fg']}44;
                            border-radius:4px;
                            padding:1px 7px;
                            font-size:0.72rem;
                            font-weight:600;
                        ">{card['badge']}</span>
                    </div>
                    <p style="font-size:0.83rem;color:#4a5568;margin:0;line-height:1.55">
                        {card['desc']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button(f"→ Mở {card['title']}", key=f"card_{card['page']}", use_container_width=True):
                _nav_to(card["page"])

    st.divider()

    # ── Section 4: Sample Questions ────────────────────────────────────────────
    st.markdown("### 💬 Câu hỏi mẫu")
    st.caption("Click **→ 🔍** để gửi câu hỏi trực tiếp tới trang GraphRAG")

    tabs = st.tabs(list(_SAMPLE_QUESTIONS.keys()))
    for tab, (category, questions) in zip(tabs, _SAMPLE_QUESTIONS.items()):
        with tab:
            for q in questions:
                col_q, col_btn = st.columns([7, 1])
                col_q.markdown(
                    f"<div style='padding:6px 0;font-size:0.92rem;color:#2d3748'>{q}</div>",
                    unsafe_allow_html=True,
                )
                if col_btn.button(
                    "→ 🔍",
                    key=f"q_{hash(q)}",
                    help=f"Gửi câu hỏi này tới GraphRAG: {q}",
                    use_container_width=True,
                ):
                    _nav_to("🔍 GraphRAG", prefill_key="graphrag_prefill", prefill_val=q)

    st.divider()

    # ── Section 5: Quick Start ─────────────────────────────────────────────────
    st.markdown("### 🚀 Hướng dẫn bắt đầu nhanh")

    _STEPS = [
        (
            "1️⃣",
            "Chọn trang **🔍 GraphRAG**",
            "Trang chính của hệ thống — hỏi đáp dựa trên Knowledge Graph Chèo.",
        ),
        (
            "2️⃣",
            "Đặt câu hỏi về Chèo",
            "Hỏi về nhân vật, vở kịch, diễn viên, trích đoạn — hoặc click câu mẫu ở trên.",
        ),
        (
            "3️⃣",
            "Xem chi tiết retrieval",
            'Mở tab <b>🕸️ Đồ thị</b> để thấy các node và quan hệ nào trong Knowledge Graph được dùng.',
        ),
        (
            "4️⃣",
            "So sánh với RAG & Chat",
            "Thử cùng câu hỏi đó trên trang <b>📚 RAG</b> và <b>💬 Chat</b> để thấy sự khác biệt.",
        ),
    ]

    step_cols = st.columns(4)
    for col, (num, title, desc) in zip(step_cols, _STEPS):
        col.markdown(
            f"""
            <div style="
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1.2rem 0.9rem;
                text-align: center;
                background: #fafbfc;
                height: 100%;
            ">
                <div style="font-size:2rem;margin-bottom:0.4rem;line-height:1">{num}</div>
                <div style="font-weight:700;font-size:0.88rem;margin-bottom:0.5rem;color:#1a202c">{title}</div>
                <div style="font-size:0.8rem;color:#6b7280;line-height:1.5">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "💡 **Mẹo sử dụng:**  \n"
        "• Hỏi **tên cụ thể** (nhân vật, vở, diễn viên) để hệ thống tìm đúng trong graph  \n"
        "• Bật **'Hiện chi tiết retrieval'** ở sidebar khi dùng GraphRAG để thấy đồ thị tri thức  \n"
        "• Trang **📊 Benchmark** cần Neo4j đang chạy và Vector Store đã build (trang 📚 RAG)"
    )
