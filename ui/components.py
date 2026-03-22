"""
Shared Streamlit UI components reused across multiple pages.
"""
from __future__ import annotations

import streamlit as st


def render_retrieval_detail(item: dict) -> None:
    """Render a 6-tab panel showing full retrieval details for one result.

    The *item* dict may come from either the GraphRAG chat history or a
    :class:`~benchmark.runner.CaseResult` ``retrieval_detail`` field.

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

    tab_stats, tab_query, tab_nodes, tab_triplets, tab_paths, tab_context = st.tabs([
        "📈 Thống kê",
        "🔍 Query",
        f"🗂️ Nodes ({num_nodes})",
        f"🔗 Triplets ({num_triplets})",
        f"🛤️ Paths ({num_paths})",
        "📝 Context (LLM input)",
    ])

    # ── Stats ─────────────────────────────────────────────────────────────────
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
