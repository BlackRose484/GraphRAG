"""
🔗 Neo4j page — kiểm tra kết nối và thông tin graph.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def render() -> None:
    st.title("🔗 Neo4j — Kiểm tra kết nối")

    from src.core.settings import settings

    with st.expander("⚙️ Cấu hình kết nối", expanded=True):
        col1, col2 = st.columns(2)
        col1.markdown(f"**URI:** `{settings.neo4j.uri}`")
        col1.markdown(f"**User:** `{settings.neo4j.user}`")
        col2.markdown(
            f"**Password:** "
            f"{'✅ Đã cấu hình' if settings.neo4j.password else '❌ Chưa cấu hình'}"
        )
        col2.markdown(f"**Ontology file (mặc định):** `{settings.ontology.file_path.name}`")

    st.divider()

    # ── Connection check ──────────────────────────────────────────────────────
    if st.button("🔍 Kiểm tra kết nối", use_container_width=False):
        from src.graph_loader.neo4j_client import Neo4jClient

        client = Neo4jClient()
        try:
            with st.spinner("Đang ping Neo4j..."):
                client.ping()
            st.success("✅ Kết nối Neo4j thành công!")

            with st.spinner("Đang lấy thông tin graph..."):
                schema = client.get_schema_summary()

            m1, m2, m3 = st.columns(3)
            m1.metric("Tổng nodes",         schema["total_nodes"])
            m2.metric("Tổng relationships", schema["total_relationships"])
            m3.metric("Node labels",        len(schema["node_labels"]))

            if schema["node_labels"]:
                st.markdown(
                    "**Labels:** "
                    + ", ".join(f"`{lb}`" for lb in sorted(schema["node_labels"]))
                )
            if schema["relationship_types"]:
                st.markdown(
                    "**Relationship types:** "
                    + ", ".join(f"`{r}`" for r in sorted(schema["relationship_types"]))
                )
        except Exception as exc:
            st.error(f"❌ {exc}")
        finally:
            client.close()

    st.divider()

    # ── Load ontology ─────────────────────────────────────────────────────────
    st.subheader("📥 Load Ontology vào Neo4j")

    ttl_files = sorted(_DATA_DIR.glob("*.ttl"))
    if not ttl_files:
        st.warning(f"Không tìm thấy file `.ttl` nào trong `{_DATA_DIR}`.")
        return

    ttl_names = [f.name for f in ttl_files]
    chosen_name = st.selectbox(
        "Chọn file ontology (.ttl)",
        options=ttl_names,
        index=0,
        help=f"Các file trong thư mục `data/`",
    )
    chosen_path = _DATA_DIR / chosen_name

    col_opt1, col_opt2 = st.columns(2)
    clear_first = col_opt1.checkbox(
        "🗑️ Xóa toàn bộ dữ liệu cũ trước khi load",
        value=False,
        help="Chạy MATCH (n) DETACH DELETE n trước — dùng khi muốn load lại từ đầu",
    )

    if col_opt2.button("▶️ Bắt đầu Load", type="primary", use_container_width=True):
        from src.graph_loader.neo4j_loader import Neo4jLoader

        with st.status("Đang load ontology…", expanded=True) as status:
            try:
                loader = Neo4jLoader()

                if clear_first:
                    st.write("🗑️ Đang xóa dữ liệu cũ…")
                    loader.clear()
                    st.write("✅ Đã xóa xong.")

                st.write(f"📖 Đang parse và load `{chosen_name}`…")
                result = loader.load(str(chosen_path))

                if result.success:
                    status.update(label="✅ Load thành công!", state="complete", expanded=True)
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Nodes created",    result.nodes_created)
                    r2.metric("Relationships",    result.relationships_created)
                    r3.metric("RDF triples",      result.triples_parsed)
                    r4.metric("Skipped",          result.skipped_nodes)
                else:
                    status.update(label=f"⚠️ Hoàn thành với {len(result.errors)} lỗi", state="error")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Nodes created",    result.nodes_created)
                    r2.metric("Relationships",    result.relationships_created)
                    r3.metric("RDF triples",      result.triples_parsed)
                    r4.metric("Skipped",          result.skipped_nodes)
                    with st.expander("❌ Chi tiết lỗi"):
                        for err in result.errors:
                            st.write(f"- {err}")

            except Exception as exc:
                status.update(label=f"❌ {exc}", state="error")
                st.exception(exc)
