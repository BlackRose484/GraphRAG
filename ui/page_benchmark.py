"""
📊 Benchmark page — evaluate GraphRAG and/or RAG against CheoBench dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

_DATASET_DIR  = Path(__file__).resolve().parents[1] / "benchmark" / "datasets"
_DEFAULT_DS   = _DATASET_DIR / "CheoBench_100_Fixed.json"
_RAG_STORE    = Path(__file__).resolve().parents[1] / "data" / "vector_store.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_registry(enabled_names: List[str]):
    """Build a MetricRegistry with only *enabled_names* active."""
    from benchmark.metrics import MetricRegistry
    registry = MetricRegistry.default()
    # start everything off, then flip on what user selected
    for m in registry.all_metrics():
        registry.set_enabled(m.name, m.name in enabled_names)
    return registry


def _get_graphrag_pipeline():
    """Cache GraphRAGPipeline in session_state."""
    key = "bm_graphrag_pipeline"
    if key not in st.session_state:
        from src.graph_loader.neo4j_client import Neo4jClient
        from src.pipeline.pipeline import GraphRAGPipeline
        client = Neo4jClient()
        client.ping()
        st.session_state[key] = GraphRAGPipeline(client)
    return st.session_state[key]


def _get_rag_pipeline():
    """Cache VectorRAGPipeline in session_state."""
    key = "bm_rag_pipeline"
    if key not in st.session_state:
        from src.rag.pipeline import VectorRAGPipeline
        st.session_state[key] = VectorRAGPipeline(store_path=_RAG_STORE)
    return st.session_state[key]


# ── Sidebar: metric selector ──────────────────────────────────────────────────

def _render_metric_selector() -> List[str]:
    """Render grouped metric toggles in sidebar. Returns list of enabled names."""
    from benchmark.metrics import MetricRegistry, MetricGroup

    registry = MetricRegistry.default()
    enabled: List[str] = []

    groups = [
        (MetricGroup.IR,    "📐 IR Metrics",     True),
        (MetricGroup.NLG,   "✍️ NLG Metrics",    True),
        (MetricGroup.EXACT, "🎯 Exact Metrics",  True),
        (MetricGroup.RAGAS, "🤖 RAGAS (chậm)",   False),
    ]

    for group, label, default_on in groups:
        metrics = registry.by_group(group)
        all_names = [m.name for m in metrics]

        # Group-level toggle
        grp_on = st.sidebar.toggle(label, value=default_on, key=f"grp_{group.value}")

        if grp_on:
            with st.sidebar.expander(f"  Chọn từng chỉ số ({group.value})", expanded=False):
                for m in metrics:
                    default_metric = group != MetricGroup.RAGAS
                    on = st.checkbox(m.name, value=default_metric, key=f"m_{m.name}")
                    if on:
                        enabled.append(m.name)
        # (if group toggled off, none of its metrics are enabled)

    return enabled


# ── Results rendering ─────────────────────────────────────────────────────────

def _render_summary(results):
    """Render side-by-side summary table and bar chart for all pipelines."""
    import pandas as pd

    rows = []
    for br in results:
        row = {"Pipeline": br.pipeline}
        row.update({
            k: (f"{v:.4f}" if v is not None else "N/A")
            for k, v in br.averages.items()
        })
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("Pipeline")

    st.subheader("📊 Kết quả trung bình")
    st.dataframe(df, use_container_width=True)

    # Bar chart — numeric averages only
    numeric_rows = []
    for br in results:
        for metric, val in br.averages.items():
            if val is not None:
                numeric_rows.append({"Pipeline": br.pipeline, "Metric": metric, "Score": val})

    if numeric_rows:
        import pandas as pd
        df_chart = pd.DataFrame(numeric_rows)
        st.subheader("📈 Biểu đồ so sánh")
        st.bar_chart(df_chart.pivot(index="Metric", columns="Pipeline", values="Score"))


def _render_case_table(results):
    """Render per-case expandable results with full retrieval detail."""
    import pandas as pd
    from ui.components import render_retrieval_detail

    st.subheader("🔍 Chi tiết từng câu hỏi")
    for br in results:
        st.markdown(f"#### {br.pipeline} — {br.n_cases} câu")

        # Summary table for quick scan
        rows = []
        for c in br.cases:
            row = {
                "ID":       c.case_id,
                "Question": c.question[:80] + "…" if len(c.question) > 80 else c.question,
                "Latency":  f"{c.latency_s:.2f}s",
                "Error":    c.error or "",
            }
            row.update({k: (f"{v:.3f}" if v is not None else "—") for k, v in c.scores.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Per-case drill-down
        st.markdown("**Xem chi tiết từng câu:**")
        for c in br.cases:
            q_short = c.question[:70] + "…" if len(c.question) > 70 else c.question
            label = f"🔎 [{c.case_id}] {q_short}"
            with st.expander(label, expanded=False):
                # Answer vs reference
                col_a, col_r = st.columns(2)
                with col_a:
                    st.markdown("**📤 Câu trả lời pipeline**")
                    st.markdown(c.answer or "_(trống)_")
                with col_r:
                    st.markdown("**📋 Ground truth**")
                    st.markdown(c.reference or "_(không có)_")

                if c.error:
                    st.error(f"❌ Lỗi: {c.error}")

                # Scores
                if c.scores:
                    score_cols = st.columns(min(len(c.scores), 4))
                    for i, (name, val) in enumerate(c.scores.items()):
                        score_cols[i % len(score_cols)].metric(
                            name,
                            f"{val:.3f}" if val is not None else "N/A"
                        )

                # Retrieval detail tabs
                if c.retrieval_detail:
                    st.divider()
                    detail_item = dict(c.retrieval_detail)
                    detail_item["latency_s"] = c.latency_s
                    render_retrieval_detail(detail_item)
                else:
                    st.caption("Không có dữ liệu retrieval chi tiết.")


def _export_buttons(results):
    """Render CSV and JSON export buttons."""
    st.subheader("📥 Xuất dữ liệu")
    col1, col2 = st.columns(2)

    # JSON export
    payload = {"results": [r.to_dict() for r in results], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    col1.download_button(
        "⬇️ Export JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # CSV export (averages only)
    try:
        import pandas as pd
        rows = []
        for br in results:
            row = {"pipeline": br.pipeline}
            row.update(br.averages)
            rows.append(row)
        csv_data = pd.DataFrame(rows).to_csv(index=False)
        col2.download_button(
            "⬇️ Export CSV",
            data=csv_data,
            file_name=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception:
        pass


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.title("📊 Benchmark — Đánh giá hệ thống RAG")
    st.caption("So sánh GraphRAG và RAG trên bộ CheoBench. Chọn chỉ số và chạy.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Cấu hình")

    # Pipeline selection
    run_graphrag = st.sidebar.toggle("🔍 Chạy GraphRAG", value=True)
    run_rag      = st.sidebar.toggle("📚 Chạy RAG",      value=True)

    # Dataset
    ds_files = sorted(_DATASET_DIR.glob("*.json"))
    ds_names = [f.name for f in ds_files]
    if ds_names:
        chosen_ds = st.sidebar.selectbox("Dataset", ds_names, index=0)
        dataset_path = _DATASET_DIR / chosen_ds
    else:
        st.sidebar.error("Không tìm thấy file dataset trong benchmark/datasets/")
        return

    n_cases = st.sidebar.slider("Số câu hỏi", min_value=1, max_value=100, value=20, step=5)

    st.sidebar.divider()
    st.sidebar.markdown("### 📐 Chọn chỉ số")
    enabled_metrics = _render_metric_selector()

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Xóa kết quả cũ"):
        st.session_state.pop("bm_results", None)
        st.rerun()

    # ── Guards ────────────────────────────────────────────────────────────────
    if not run_graphrag and not run_rag:
        st.warning("Chọn ít nhất một pipeline ở sidebar.")
        return
    if not enabled_metrics:
        st.warning("Chọn ít nhất một chỉ số ở sidebar.")
        return
    if run_rag and not _RAG_STORE.exists():
        st.error(
            "Vector store của RAG chưa được build. "
            "Vào trang **📚 RAG** → Build / Rebuild Vector Store trước."
        )
        return

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("▶️ Chạy Benchmark", type="primary", use_container_width=True):
        from benchmark.runner import BenchmarkRunner

        registry = _load_registry(enabled_metrics)

        graphrag_pipe = None
        rag_pipe      = None

        with st.status("Đang khởi tạo pipeline…", expanded=True) as status:
            try:
                if run_graphrag:
                    st.write("Kết nối Neo4j + khởi tạo GraphRAGPipeline…")
                    graphrag_pipe = _get_graphrag_pipeline()
                if run_rag:
                    st.write("Tải VectorRAGPipeline…")
                    rag_pipe = _get_rag_pipeline()

                runner = BenchmarkRunner(registry=registry)

                total_steps = n_cases * sum([run_graphrag, run_rag])
                progress    = st.progress(0, text="Bắt đầu …")

                _step_counter = {"n": 0}

                def _progress_cb(current: int, total: int, msg: str) -> None:
                    _step_counter["n"] = current
                    pct = current / total if total else 0
                    progress.progress(pct, text=f"[{current}/{total}] {msg}")

                st.write(f"Chạy {n_cases} câu trên {dataset_path.name} …")
                t0      = time.time()
                results = runner.run(
                    dataset_path=dataset_path,
                    graphrag_pipeline=graphrag_pipe,
                    rag_pipeline=rag_pipe,
                    n_cases=n_cases,
                    progress_cb=_progress_cb,
                )
                elapsed = time.time() - t0
                progress.progress(1.0, text="✅ Hoàn thành!")
                status.update(
                    label=f"✅ Hoàn thành {n_cases} câu trong {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )
                st.session_state["bm_results"] = results

            except Exception as exc:
                status.update(label=f"❌ {exc}", state="error")
                st.exception(exc)

    # ── Display results ───────────────────────────────────────────────────────
    results = st.session_state.get("bm_results")
    if results:
        st.divider()
        _render_summary(results)
        st.divider()
        _render_case_table(results)
        st.divider()
        _export_buttons(results)
