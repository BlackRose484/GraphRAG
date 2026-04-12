"""
Shared Streamlit UI components reused across multiple pages.
"""
from __future__ import annotations

import streamlit as st

try:
    from ui.graph_visualizer import build_graph_html as _build_graph_html, is_available as _gviz_available
except ImportError:
    _gviz_available = lambda: False
    _build_graph_html = None


def render_retrieval_detail(item: dict, key_prefix: str = "", compact: bool = False) -> None:
    """Render a retrieval detail panel showing full retrieval details for one result.

    The *item* dict may come from either the GraphRAG chat history or a
    :class:`~benchmark.runner.CaseResult` ``retrieval_detail`` field.

    Args:
        item:       The result dict to render.
        key_prefix: Prefix for all Streamlit widget keys — use when rendering
                    the same component multiple times on a page to avoid key
                    conflicts (e.g. chat history, benchmark rows).
        compact:    When True, uses shorter tab labels and hides Stats/Graph tabs.
                    Set to True when rendering inside narrow columns (e.g. Compare page).

    Expected keys (all optional, falls back gracefully):
        - num_nodes / num_triplets / num_paths  (int)
        - retrieval_time / gen_time / total_time  (float)  — OR —
          latency_s (float, used when individual times not available)
        - processed_query   (dict: original, expanded, decomposed)
        - nodes             (list of dicts)
        - triplets          (list of 3-tuples or lists)
        - paths             (list of lists/strings)
        - formatted_contexts (dict str → str)
    """
    r = item  # alias

    num_nodes    = r.get("num_nodes",    len(r.get("nodes",    [])))
    num_triplets = r.get("num_triplets", len(r.get("triplets", [])))
    num_paths    = r.get("num_paths",    len(r.get("paths",    [])))

    # compact=True: short labels, no Stats/Graph tabs (for narrow columns e.g. Compare page)
    if compact:
        tab_query, tab_nodes, tab_triplets, tab_paths, tab_context = st.tabs([
            "🔍 Query",
            f"Nodes ({num_nodes})",
            f"Triplets ({num_triplets})",
            f"Paths ({num_paths})",
            "📝 Context",
        ])
        tab_stats = None
        tab_graph = None
    else:
        tab_stats, tab_query, tab_nodes, tab_triplets, tab_paths, tab_context, tab_graph = st.tabs([
            "📈 Thống kê",
            "🔍 Query",
            f"🗂️ Nodes ({num_nodes})",
            f"🔗 Triplets ({num_triplets})",
            f"🛤️ Paths ({num_paths})",
            "📝 Context (LLM input)",
            "🕸️ Đồ thị",
        ])

    # ── Stats ─────────────────────────────────────────────────────────────────
    if tab_stats is not None:
      with tab_stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes",    num_nodes)
        c2.metric("Triplets", num_triplets)
        c3.metric("Paths",    num_paths)

        # Timing — support both chat-page and benchmark-page item shapes
        if "retrieval_time" in r:
            st.caption(
                f"⏱ Retrieval: **{r['retrieval_time']:.2f}s** | "
                f"Generation: **{r.get('gen_time', 0):.2f}s** | "
                f"Total: **{r.get('total_time', 0):.2f}s**"
            )
        elif "latency_s" in r:
            st.caption(f"⏱ Latency: **{r['latency_s']:.2f}s**")

    # ── Query processing ──────────────────────────────────────────────────────
    with tab_query:
        pq = r.get("processed_query", {})
        if pq:
            st.markdown("**📝 Câu hỏi gốc**")
            st.info(pq.get("original", "—"))

            expanded = pq.get("expanded", "")
            if expanded and expanded != pq.get("original"):
                st.markdown("**✨ Câu hỏi mở rộng (expanded)**")
                st.success(expanded)

            decomposed = pq.get("decomposed", [])
            if decomposed and decomposed != [pq.get("original")]:
                st.markdown(f"**🔀 Sub-queries ({len(decomposed)})**")
                for i, sq in enumerate(decomposed, 1):
                    st.write(f"{i}. {sq}")
        else:
            st.info("Không có thông tin query processing.")

        # ── Entities extracted ─────────────────────────────────────────────
        entities = r.get("entities", {})
        _ENTITY_CONFIG = {
            "characters": ("🎭", "Nhân vật",  "#1f77b4"),
            "actors":     ("🎤", "Diễn viên", "#2ca02c"),
            "plays":      ("📖", "Vở kịch",   "#d62728"),
            "scenes":     ("🎬", "Trích đoạn", "#9467bd"),
        }
        has_entities = any(entities.get(k) for k in _ENTITY_CONFIG)
        if has_entities:
            st.markdown("---")
            st.markdown("**🔍 Entities đã trích xuất**")
            cols = st.columns(4)
            for col, (key, (icon, label, color)) in zip(cols, _ENTITY_CONFIG.items()):
                names = entities.get(key, [])
                with col:
                    st.markdown(
                        f"<div style='font-size:0.8em;color:gray'>{icon} {label}</div>",
                        unsafe_allow_html=True,
                    )
                    if names:
                        for name in names:
                            st.markdown(
                                f"<span style='background:{color}22;border:1px solid {color}66;"
                                f"border-radius:4px;padding:2px 7px;margin:2px;display:inline-block;"
                                f"font-size:0.85em'>{name}</span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "<span style='color:gray;font-size:0.8em'>—</span>",
                            unsafe_allow_html=True,
                        )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    with tab_nodes:
        nodes = r.get("nodes", [])
        if nodes:
            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(nodes), use_container_width=True)
            except Exception:
                for i, node in enumerate(nodes):
                    label = (
                        node.get("name")
                        or node.get("charName")
                        or node.get("title")
                        or str(node)[:60]
                    )
                    with st.expander(f"Node {i + 1}: {label}"):
                        st.json(node)
        else:
            st.info("Không có node nào được truy xuất.")

    # ── Triplets ──────────────────────────────────────────────────────────────
    with tab_triplets:
        triplets = r.get("triplets", [])
        if triplets:
            try:
                import pandas as pd
                rows = []
                for t in triplets:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        rows.append({"Subject": t[0], "Relation": t[1], "Object": t[2]})
                    else:
                        rows.append({"Subject": str(t), "Relation": "", "Object": ""})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception:
                for t in triplets:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        st.write(f"**{t[0]}** —[{t[1]}]→ **{t[2]}**")
                    else:
                        st.write(str(t))
        else:
            st.info("Không có triplet nào được truy xuất.")

    # ── Paths ─────────────────────────────────────────────────────────────────
    with tab_paths:
        paths = r.get("paths", [])
        if paths:
            for i, path in enumerate(paths, 1):
                if isinstance(path, (list, tuple)):
                    st.write(f"**Path {i}:** " + " → ".join(str(p) for p in path))
                else:
                    st.write(f"**Path {i}:** {path}")
        else:
            st.info("Không có path nào được truy xuất.")

    # ── Context ───────────────────────────────────────────────────────────────
    with tab_context:
        fmts = r.get("formatted_contexts", {})
        if fmts:
            fmt_keys = list(fmts.keys())
            if len(fmt_keys) == 1:
                # Single format — show directly
                st.caption(f"**{fmt_keys[0]}**")
                st.text_area(
                    label=fmt_keys[0],
                    value=fmts[fmt_keys[0]],
                    height=350,
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                # Multiple formats — use tabs (allowed inside expanders)
                fmt_tabs = st.tabs([f"📄 {k}" for k in fmt_keys])
                for tab, key in zip(fmt_tabs, fmt_keys):
                    with tab:
                        st.text_area(
                            label=key,
                            value=fmts[key],
                            height=350,
                            disabled=True,
                            label_visibility="collapsed",
                        )
        else:
            st.info("Không có formatted context.")

    # ── Graph visualization ────────────────────────────────────────────────
    if tab_graph is not None:
      with tab_graph:
        nodes    = r.get("nodes", [])
        triplets = r.get("triplets", [])
        subgraph = r.get("subgraph", {})

        if not nodes and not triplets:
            st.info("Đồ thị trống — cháy pipeline trước để xem kết quả.")
        elif not _gviz_available():
            st.warning("⚠️ pyvis chưa cài. Chạy: `pip install pyvis`")
        else:
            # Controls
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                height = st.slider(
                    "Chiều cao đồ thị (px)",
                    min_value=300, max_value=900, value=600, step=50,
                    key=f"{key_prefix}graph_height",
                )
            with col2:
                max_nodes = st.slider(
                    "Số node tối đa hiển thị",
                    min_value=20, max_value=120, value=80, step=10,
                    key=f"{key_prefix}graph_max_nodes",
                )
            with col3:
                physics = st.toggle("Physics", value=True, key=f"{key_prefix}graph_physics")

            st.caption(
                f"💡 **{len(nodes)} nodes** + **{len(triplets)} triplets** từ retrieval. "
                "Kéo để di chuyển, cuộn để phóng to/thu nhỏ, hover để xem chi tiết."
            )

            # Build and render
            html = _build_graph_html(
                nodes=nodes[:max_nodes],
                triplets=triplets,
                subgraph=subgraph,
                height=height,
                physics=physics,
            )
            st.components.v1.html(html, height=height + 30, scrolling=False)

            # Legend
            st.markdown(
                """
                <div style='display:flex;flex-wrap:wrap;gap:14px;font-size:0.82em;margin-top:6px'>
                  <span><span style='color:#e8811a;font-size:1.2em'>■</span> Vở chèo</span>
                  <span><span style='color:#2e7dd6;font-size:1.2em'>●</span> Nhân vật</span>
                  <span><span style='color:#1a9c4e;font-size:1.2em'>◆</span> Diễn viên</span>
                  <span><span style='color:#7b3fa0;font-size:1.2em'>●</span> Trích đoạn</span>
                  <span><span style='color:#5bafd6;font-size:1.2em'>▼</span> Phiên bản</span>
                  <span><span style='color:#c0c0c0;font-size:1.2em'>■</span> RoleAssignment</span>
                  <span><span style='color:#f0c040;font-size:1.2em'>★</span> Appearance</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

