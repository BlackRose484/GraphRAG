"""
📊 Benchmark page — evaluate GraphRAG and/or RAG against CheoBench dataset.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

_DATASET_DIR    = Path(__file__).resolve().parents[1] / "benchmark" / "datasets"
_DEFAULT_DS     = _DATASET_DIR / "CheoBench_v2.json"
_RAG_STORE      = Path(__file__).resolve().parents[1] / "data" / "vector_store.pkl"
_AUTO_RUNS_DIR  = Path(__file__).resolve().parents[1] / "benchmark" / "results" / "auto_benchmark"


# ── Dataset helpers ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_dataset_index(path: str) -> List[Dict]:
    """Load dataset and return [{id, category, question}, ...] for selectors."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("test_cases", data) if isinstance(data, dict) else data
    return [
        {
            "id":       c.get("id", f"CASE_{i+1:03d}"),
            "category": c.get("category", "uncategorized"),
            "question": c.get("question", ""),
        }
        for i, c in enumerate(raw)
        if c.get("question")
    ]


def _select_case_ids(dataset_path: Path) -> Tuple[Optional[List[str]], int]:
    """Render question-selection UI in sidebar. Returns (case_ids_or_None, count)."""
    index = _load_dataset_index(str(dataset_path))
    total_available = len(index)
    if total_available == 0:
        st.sidebar.error("Dataset rỗng.")
        return [], 0

    mode = st.sidebar.radio(
        "Chế độ chọn câu hỏi",
        ["🎲 Ngẫu nhiên", "✋ Tự chọn"],
        horizontal=True,
        key="bm_select_mode",
    )

    if mode == "🎲 Ngẫu nhiên":
        n = st.sidebar.slider(
            "Số câu hỏi",
            min_value=1,
            max_value=total_available,
            value=min(20, total_available),
            step=1,
        )
        seed_str = st.sidebar.text_input(
            "Random seed (để trống = ngẫu nhiên thật)",
            value="42",
            help="Đặt seed cố định để 2 lần chạy cùng N câu chọn ra cùng tập câu hỏi (reproducible).",
        )
        seed: Optional[int]
        try:
            seed = int(seed_str) if seed_str.strip() else None
        except ValueError:
            seed = None
            st.sidebar.warning("Seed không hợp lệ, dùng ngẫu nhiên thật.")

        # Sample case IDs
        rng = random.Random(seed)
        all_ids = [c["id"] for c in index]
        chosen = sorted(rng.sample(all_ids, n))   # sort for stable display
        with st.sidebar.expander(f"  Đã chọn {n} câu", expanded=False):
            id_to_q = {c["id"]: c["question"] for c in index}
            for cid in chosen:
                q = id_to_q.get(cid, "")
                st.caption(f"`{cid}` — {q[:60]}{'…' if len(q) > 60 else ''}")
        return chosen, n

    # ── Manual selection mode ─────────────────────────────────────────────────
    # Filter by category first to make multiselect manageable
    categories = sorted({c["category"] for c in index})
    cat_pick = st.sidebar.multiselect(
        "Lọc theo category",
        categories,
        default=categories,
        help="Chọn category(s) để giới hạn danh sách câu hỏi bên dưới.",
    )
    filtered = [c for c in index if c["category"] in set(cat_pick)] if cat_pick else index

    # Build display labels: "CASE_001 — local — Mô tả Thị Kính?"
    labels: List[str] = []
    label_to_id: Dict[str, str] = {}
    for c in filtered:
        q = c["question"]
        q_short = q[:80] + "…" if len(q) > 80 else q
        label = f"{c['id']} | {c['category']} | {q_short}"
        labels.append(label)
        label_to_id[label] = c["id"]

    chosen_labels = st.sidebar.multiselect(
        f"Chọn câu hỏi ({len(filtered)} sẵn có)",
        labels,
        default=[],
        help="Chọn cụ thể các câu hỏi muốn benchmark.",
    )
    chosen_ids = [label_to_id[l] for l in chosen_labels]
    if chosen_ids:
        st.sidebar.success(f"✅ Đã chọn {len(chosen_ids)} câu")
    return chosen_ids, len(chosen_ids)


# ── History (load past auto-saved runs) ───────────────────────────────────────

def _list_history_runs() -> List[Path]:
    """Return list of timestamped run directories under auto_benchmark/."""
    if not _AUTO_RUNS_DIR.exists():
        return []
    return sorted(
        [p for p in _AUTO_RUNS_DIR.iterdir() if p.is_dir()],
        reverse=True,  # newest first
    )


def _load_run_from_disk(run_dir: Path):
    """Reconstruct ``List[BenchmarkResult]`` from an on-disk run.

    Prefers ``final.json`` (complete run). Falls back to ``partial.jsonl`` so
    a crashed run can still be inspected.
    """
    from benchmark.runner import BenchmarkResult, CaseResult

    final = run_dir / "final.json"
    if final.exists():
        data = json.loads(final.read_text(encoding="utf-8"))
        out = []
        for br_dict in data.get("results", []):
            cases = [
                CaseResult(
                    case_id=c["case_id"], question=c["question"],
                    pipeline=br_dict["pipeline"], answer=c["answer"],
                    reference=c["reference"], category=c.get("category", ""),
                    scores=c.get("scores", {}), latency_s=c.get("latency_s", 0.0),
                    error=c.get("error"),
                )
                for c in br_dict.get("cases", [])
            ]
            out.append(BenchmarkResult(
                pipeline=br_dict["pipeline"],
                n_cases=br_dict["n_cases"],
                cases=cases,
                averages=br_dict.get("averages", {}),
                by_category=br_dict.get("by_category", {}),
            ))
        return out

    partial = run_dir / "partial.jsonl"
    if partial.exists():
        from benchmark.runner import BenchmarkRunner
        cases_by_pipe: Dict[str, List] = {}
        for line in partial.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            cases_by_pipe.setdefault(d["pipeline"], []).append(
                CaseResult(
                    case_id=d["case_id"], question=d["question"],
                    pipeline=d["pipeline"], answer=d["answer"],
                    reference=d["reference"], category=d.get("category", ""),
                    scores=d.get("scores", {}), latency_s=d.get("latency_s", 0.0),
                    error=d.get("error"),
                )
            )
        out = []
        for pipe, cases in cases_by_pipe.items():
            out.append(BenchmarkResult(
                pipeline=pipe,
                n_cases=len(cases),
                cases=cases,
                averages=BenchmarkRunner._aggregate(cases),
                by_category=BenchmarkRunner._aggregate_by_category(cases),
            ))
        return out

    return None


def _render_history_loader() -> None:
    runs = _list_history_runs()
    if not runs:
        st.sidebar.caption("_(Chưa có run nào được lưu)_")
        return

    options = {p.name: p for p in runs}
    chosen = st.sidebar.selectbox(
        f"Chọn run để load ({len(runs)} run)",
        ["—"] + list(options.keys()),
        key="bm_history_pick",
    )
    if chosen != "—" and st.sidebar.button("📥 Load run này"):
        try:
            loaded = _load_run_from_disk(options[chosen])
            if loaded:
                st.session_state["bm_results"]    = loaded
                st.session_state["bm_output_dir"] = str(options[chosen])
                st.sidebar.success(f"Đã load `{chosen}`")
                st.rerun()
            else:
                st.sidebar.error("Run này không có data hợp lệ.")
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")


# ── Pipeline / registry helpers ───────────────────────────────────────────────

def _load_registry(enabled_names: List[str]):
    """Build a MetricRegistry with only *enabled_names* active."""
    from benchmark.metrics import MetricRegistry
    registry = MetricRegistry.default()
    for m in registry.all_metrics():
        registry.set_enabled(m.name, m.name in enabled_names)
    return registry


def _get_graphrag_pipeline():
    key = "bm_graphrag_pipeline"
    if key not in st.session_state:
        from src.graph_loader.neo4j_client import Neo4jClient
        from src.pipeline.pipeline import GraphRAGPipeline
        client = Neo4jClient()
        client.ping()
        st.session_state[key] = GraphRAGPipeline(client)
    return st.session_state[key]


def _get_rag_pipeline():
    key = "bm_rag_pipeline"
    if key not in st.session_state:
        from src.rag.pipeline import VectorRAGPipeline
        st.session_state[key] = VectorRAGPipeline(store_path=_RAG_STORE)
    return st.session_state[key]


# ── Sidebar: metric selector ──────────────────────────────────────────────────

def _render_metric_selector() -> List[str]:
    from benchmark.metrics import MetricRegistry, MetricGroup

    registry = MetricRegistry.default()
    enabled: List[str] = []

    # Default-on follows the thesis spec composite: IR + RAGAS are required
    # for S_retrieval / S_generation; Exact metrics are off (literal-match,
    # superseded by RAGAs Faithfulness / AnswerRelevance).
    groups = [
        (MetricGroup.IR,    "📐 IR Metrics",          True),
        (MetricGroup.RAGAS, "🤖 RAGAS (LLM-judged)",  True),
        (MetricGroup.EXACT, "🎯 Exact Metrics (debug, off mặc định)", False),
    ]

    for group, label, default_on in groups:
        metrics = registry.by_group(group)
        if not metrics:
            continue
        grp_on = st.sidebar.toggle(label, value=default_on, key=f"grp_{group.value}")
        if grp_on:
            with st.sidebar.expander(f"  Chọn từng chỉ số ({group.value})", expanded=False):
                for m in metrics:
                    default_metric = registry.is_enabled(m.name)
                    on = st.checkbox(m.name, value=default_metric, key=f"m_{m.name}")
                    if on:
                        enabled.append(m.name)

    return enabled


# ── Composite-score tiles ─────────────────────────────────────────────────────

_COMPOSITE_KEYS = ("S_retrieval", "S_generation", "S_overall")


def _render_composite_scores(results):
    """Top-of-page big tiles for the 3 composite scores per pipeline."""
    if not results:
        return
    st.subheader("🎯 Điểm tổng hợp")
    for br in results:
        st.markdown(f"**{br.pipeline}**")
        cols = st.columns(len(_COMPOSITE_KEYS))
        for i, k in enumerate(_COMPOSITE_KEYS):
            v = br.averages.get(k)
            cols[i].metric(k, f"{v:.3f}" if v is not None else "N/A")


# ── Per-category breakdown ────────────────────────────────────────────────────

def _render_by_category(results):
    """Show how each pipeline performs on local / community / global queries."""
    if not results or not any(br.by_category for br in results):
        return
    import pandas as pd

    st.subheader("📂 Phân tích theo category câu hỏi")
    st.caption(
        "GraphRAG được thiết kế cho multi-hop (community / global). "
        "RAG có lợi thế trên local queries (1-2 hop)."
    )

    # Build wide table: rows = (pipeline, category), cols = composite scores
    rows = []
    for br in results:
        for cat, scores in br.by_category.items():
            row = {
                "Pipeline": br.pipeline,
                "Category": cat,
                "N":        sum(1 for c in br.cases if c.category == cat),
            }
            for k in _COMPOSITE_KEYS:
                v = scores.get(k)
                row[k] = round(v, 3) if v is not None else None
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart: S_overall by (pipeline, category)
        try:
            chart_df = df.pivot(index="Category", columns="Pipeline", values="S_overall")
            st.bar_chart(chart_df, height=280)
        except Exception:
            pass


# ── Aggregated summary table ──────────────────────────────────────────────────

def _render_summary(results):
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
    st.subheader("📊 Trung bình toàn bộ câu hỏi")
    st.dataframe(df, use_container_width=True)

    numeric_rows = []
    for br in results:
        for metric, val in br.averages.items():
            if val is not None and metric not in _COMPOSITE_KEYS:
                numeric_rows.append({"Pipeline": br.pipeline, "Metric": metric, "Score": val})
    if numeric_rows:
        df_chart = pd.DataFrame(numeric_rows)
        st.subheader("📈 Biểu đồ so sánh chi tiết")
        st.bar_chart(df_chart.pivot(index="Metric", columns="Pipeline", values="Score"))


# ── Per-case detail ───────────────────────────────────────────────────────────

def _render_case_table(results):
    import pandas as pd
    from ui.components import render_retrieval_detail

    st.subheader("🔍 Chi tiết từng câu hỏi")
    for br in results:
        st.markdown(f"#### {br.pipeline} — {br.n_cases} câu")

        rows = []
        for c in br.cases:
            row = {
                "ID":       c.case_id,
                "Category": c.category,
                "Question": c.question[:80] + "…" if len(c.question) > 80 else c.question,
                "Latency":  f"{c.latency_s:.2f}s",
                "Error":    c.error or "",
            }
            row.update({k: (f"{v:.3f}" if v is not None else "—") for k, v in c.scores.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("**Xem chi tiết từng câu:**")
        for c in br.cases:
            q_short = c.question[:70] + "…" if len(c.question) > 70 else c.question
            label = f"🔎 [{c.case_id}] [{c.category}] {q_short}"
            with st.expander(label, expanded=False):
                col_a, col_r = st.columns(2)
                with col_a:
                    st.markdown("**📤 Câu trả lời pipeline**")
                    st.markdown(c.answer or "_(trống)_")
                with col_r:
                    st.markdown("**📋 Ground truth**")
                    st.markdown(c.reference or "_(không có)_")

                if c.error:
                    st.error(f"❌ Lỗi: {c.error}")

                if c.scores:
                    score_cols = st.columns(min(len(c.scores), 4))
                    for i, (name, val) in enumerate(c.scores.items()):
                        score_cols[i % len(score_cols)].metric(
                            name,
                            f"{val:.3f}" if val is not None else "N/A"
                        )

                if c.retrieval_detail:
                    st.divider()
                    detail_item = dict(c.retrieval_detail)
                    detail_item["latency_s"] = c.latency_s
                    render_retrieval_detail(
                        detail_item,
                        key_prefix=f"bm_{br.pipeline}_{c.case_id}_",
                    )
                else:
                    st.caption("Không có dữ liệu retrieval chi tiết.")


# ── Export ────────────────────────────────────────────────────────────────────

def _export_buttons(results):
    st.subheader("📥 Xuất dữ liệu")
    col1, col2 = st.columns(2)

    payload = {
        "results":   [r.to_dict() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    col1.download_button(
        "⬇️ Export JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    try:
        import pandas as pd
        rows = []
        for br in results:
            row = {"pipeline": br.pipeline}
            row.update(br.averages)
            rows.append(row)
        csv_data = pd.DataFrame(rows).to_csv(index=False)
        col2.download_button(
            "⬇️ Export CSV (averages)",
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
    st.caption("So sánh GraphRAG và RAG trên bộ CheoBench. Chọn câu hỏi, chỉ số, rồi chạy.")

    with st.expander("📐 Khung điểm tổng hợp (composite scoring)", expanded=False):
        from benchmark.score_aggregator import (
            RETRIEVAL_WEIGHTS, GENERATION_WEIGHTS,
            OVERALL_RETRIEVAL_WEIGHT, OVERALL_GENERATION_WEIGHT,
        )

        def _weights_md(weights: Dict[str, float]) -> str:
            lines = ["| Metric | Trọng số |", "|---|---|"]
            for k, w in weights.items():
                lines.append(f"| {k} | {w:.2f} |")
            lines.append(f"| **Tổng** | **{sum(weights.values()):.2f}** |")
            return "\n".join(lines)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### Chất lượng truy xuất ($S_{retrieval}$)")
            st.markdown(_weights_md(RETRIEVAL_WEIGHTS))
        with col_r:
            st.markdown("##### Chất lượng tạo sinh ($S_{generation}$)")
            st.markdown(_weights_md(GENERATION_WEIGHTS))
        st.markdown(
            f"##### Điểm tổng hợp\n"
            f"$S_{{overall}} = {OVERALL_RETRIEVAL_WEIGHT} \\cdot S_{{retrieval}} + "
            f"{OVERALL_GENERATION_WEIGHT} \\cdot S_{{generation}}$"
        )

    with st.expander("ℹ️ Hướng dẫn sử dụng", expanded=False):
        st.markdown(
            """
            Trang này **đánh giá định lượng** chất lượng câu trả lời của 2 hệ thống:

            | Hệ thống | Mô tả |
            |---|---|
            | 🔍 **GraphRAG** | Truy xuất từ Knowledge Graph + LLM |
            | 📚 **RAG**      | Vector search + LLM |

            **Cách dùng:**
            1. Chọn pipeline cần chạy (GraphRAG / RAG / cả hai)
            2. Chọn dataset
            3. Chọn câu hỏi: **🎲 Ngẫu nhiên** N câu, hoặc **✋ Tự chọn** từng câu
            4. Bật / tắt các nhóm chỉ số (IR, Exact, RAGAS)
            5. Nhấn **▶️ Chạy Benchmark**
            6. Xem điểm tổng hợp, phân tích theo category, và chi tiết từng câu

            > ⚠️ Cần Neo4j + Vector Store đã build trước.
            """
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Cấu hình")

    run_graphrag = st.sidebar.toggle("🔍 Chạy GraphRAG", value=True)
    run_rag      = st.sidebar.toggle("📚 Chạy RAG",      value=True)

    ds_files = sorted(_DATASET_DIR.glob("*.json"))
    ds_names = [f.name for f in ds_files]
    if not ds_names:
        st.sidebar.error("Không tìm thấy file dataset trong benchmark/datasets/")
        return
    chosen_ds    = st.sidebar.selectbox("Dataset", ds_names, index=0)
    dataset_path = _DATASET_DIR / chosen_ds

    st.sidebar.divider()
    st.sidebar.markdown("### 📋 Chọn câu hỏi")
    case_ids, n_selected = _select_case_ids(dataset_path)

    st.sidebar.divider()
    st.sidebar.markdown("### 📐 Chọn chỉ số")
    enabled_metrics = _render_metric_selector()

    st.sidebar.divider()
    st.sidebar.markdown("### 📂 Lịch sử benchmark")
    _render_history_loader()

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Xóa kết quả hiển thị"):
        st.session_state.pop("bm_results", None)
        st.session_state.pop("bm_output_dir", None)
        st.rerun()

    # ── Guards ────────────────────────────────────────────────────────────────
    if not run_graphrag and not run_rag:
        st.warning("Chọn ít nhất một pipeline ở sidebar.")
        return
    if not enabled_metrics:
        st.warning("Chọn ít nhất một chỉ số ở sidebar.")
        return
    if n_selected == 0:
        st.warning("Chưa chọn câu hỏi nào. Hãy chọn ở sidebar.")
        return
    if run_rag and not _RAG_STORE.exists():
        st.error(
            "Vector store của RAG chưa được build. "
            "Vào trang **📚 RAG** → Build / Rebuild Vector Store trước."
        )
        return

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button(
        f"▶️ Chạy Benchmark trên {n_selected} câu",
        type="primary",
        use_container_width=True,
    ):
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

                progress = st.progress(0, text="Bắt đầu …")

                def _progress_cb(current: int, total: int, msg: str) -> None:
                    pct = current / total if total else 0
                    progress.progress(pct, text=f"[{current}/{total}] {msg}")

                # Auto-save dir: timestamped folder under auto_benchmark/
                run_id     = time.strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = _AUTO_RUNS_DIR / run_id
                st.write(f"📁 Lưu tiến trình vào: `{output_dir.relative_to(Path.cwd()) if output_dir.is_relative_to(Path.cwd()) else output_dir}`")

                st.write(
                    f"Chạy {n_selected} câu trên {dataset_path.name}…"
                )
                t0      = time.time()
                results = runner.run(
                    dataset_path=dataset_path,
                    graphrag_pipeline=graphrag_pipe,
                    rag_pipeline=rag_pipe,
                    case_ids=case_ids,
                    progress_cb=_progress_cb,
                    output_dir=output_dir,
                )
                elapsed = time.time() - t0
                progress.progress(1.0, text="✅ Hoàn thành!")
                status.update(
                    label=f"✅ Hoàn thành {n_selected} câu trong {elapsed:.1f}s — đã lưu vào `{run_id}`",
                    state="complete",
                    expanded=False,
                )
                st.session_state["bm_results"]    = results
                st.session_state["bm_output_dir"] = str(output_dir)

            except Exception as exc:
                status.update(label=f"❌ {exc}", state="error")
                st.exception(exc)

    # ── Display results ───────────────────────────────────────────────────────
    results = st.session_state.get("bm_results")
    if results:
        st.divider()
        out_dir = st.session_state.get("bm_output_dir")
        if out_dir:
            st.info(f"💾 Run đã lưu tại: `{out_dir}`")
        _render_composite_scores(results)
        st.divider()
        _render_by_category(results)
        st.divider()
        _render_summary(results)
        st.divider()
        _render_case_table(results)
        st.divider()
        _export_buttons(results)
