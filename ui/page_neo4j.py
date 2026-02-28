"""
🔗 Neo4j page — kiểm tra kết nối và thông tin graph.
"""

from __future__ import annotations

import streamlit as st


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
        col2.markdown(f"**Ontology file:** `{settings.ontology.file_path.name}`")

    st.divider()

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
