"""
Batch-generate answers for all 21 experiment questions x 3 systems.

Outputs:
  - benchmark/datasets/pregenerated_answers.json (structured data + blind label mapping — read by the Preference page at runtime)
  - benchmark/results/pregenerated_answers.md    (human-readable, copy-paste for forms)

Usage:
    python -m benchmark.generate_answers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Question bank (mirrored from ui/page_experiment.py) ─────────────────────

QUESTIONS: dict[str, list[dict[str, str]]] = {
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
        {"id": "CASE_063", "q": "Diễn viên An Chinh đã đóng những vai nào?"},
    ],
    "Dạng 3 — Phân tích & So sánh (Khó)": [
        {"id": "CASE_078", "q": "Hãy liệt kê những vở chèo cổ tiêu biểu?"},
        {"id": "CASE_082", "q": "So sánh chủ đề của vở Quan Âm Thị Kính và Kim Nham?"},
        {"id": "CASE_083", "q": "So sánh nhân vật nữ chính của các vở chèo cổ?"},
        {"id": "CASE_090", "q": "Nhân vật loại Đào trong chèo là gì? Có những ai là đại diện tiêu biểu?"},
        {"id": "CASE_093", "q": "Phân tích hình tượng người phụ nữ hy sinh trong các vở chèo cổ?"},
        {"id": "CASE_094", "q": "So sánh số phận bi kịch của Súy Vân và Thị Kính?"},
        {"id": "CASE_097", "q": "Liệt kê các mối quan hệ vợ-chồng trong các vở chèo cổ và kết cục của họ?"},
    ],
}

_RAG_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "vector_store.pkl"
_DATASETS_DIR = Path(__file__).resolve().parents[0] / "datasets"
_RESULTS_DIR  = Path(__file__).resolve().parents[0] / "results" / "user_studies"

_CHAT_SYSTEM_PROMPT = (
    "Bạn là trợ lý chuyên về nghệ thuật Chèo Việt Nam. "
    "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác."
)

SYSTEMS = ["graphrag", "rag", "chat"]

# Display names for the form (hide real system identity)
DISPLAY_NAMES: dict[str, str] = {
    "chat":     "Hệ thống AI",
    "rag":      "Hệ thống AI Plus",
    "graphrag": "Hệ thống AI Pro",
}

# Order to display in the form
DISPLAY_ORDER = ["chat", "rag", "graphrag"]


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class SystemResult:
    answer: str
    elapsed: float
    error: Optional[str] = None
    metadata: dict[str, Any] | None = None


# ── Runner functions (no Streamlit dependency) ───────────────────────────────

def _run_graphrag(query: str) -> SystemResult:
    t0 = time.time()
    try:
        from src.graph_loader.neo4j_client import Neo4jClient
        from src.pipeline.pipeline import GraphRAGPipeline

        client = Neo4jClient()
        client.ping()
        pipeline = GraphRAGPipeline(client)
        result = pipeline.run(query)

        return SystemResult(
            answer=result.answer,
            elapsed=time.time() - t0,
            metadata={
                "num_nodes": result.retrieval.num_nodes,
                "num_triplets": result.retrieval.num_triplets,
                "retrieval_time": round(result.retrieval.retrieval_time, 2),
                "total_time": round(result.total_time, 2),
            },
        )
    except Exception as exc:
        return SystemResult(answer="", elapsed=time.time() - t0, error=str(exc))


def _run_rag(query: str) -> SystemResult:
    t0 = time.time()
    try:
        from src.rag.pipeline import VectorRAGPipeline

        pipeline = VectorRAGPipeline(store_path=_RAG_STORE_PATH, top_k=5)
        result = pipeline.run(query)

        if result.generation.error:
            return SystemResult(
                answer="", elapsed=time.time() - t0,
                error=result.generation.error,
            )
        return SystemResult(
            answer=result.answer,
            elapsed=time.time() - t0,
            metadata={
                "num_chunks": result.retrieval.num_nodes,
                "retrieval_time": round(result.retrieval.retrieval_time, 2),
            },
        )
    except Exception as exc:
        return SystemResult(answer="", elapsed=time.time() - t0, error=str(exc))


def _run_chat(query: str) -> SystemResult:
    t0 = time.time()
    try:
        from src.core.base import BaseModel

        class _ChatModel(BaseModel):
            pass

        model = _ChatModel()
        prompt = (
            f"{_CHAT_SYSTEM_PROMPT}\n\n"
            f"Người dùng: {query}\n"
            f"Trợ lý:"
        )
        answer = model.safe_generate(prompt)
        return SystemResult(answer=answer, elapsed=time.time() - t0)
    except Exception as exc:
        return SystemResult(answer="", elapsed=time.time() - t0, error=str(exc))


def _run_all(query: str) -> dict[str, SystemResult]:
    runners = {
        "graphrag": _run_graphrag,
        "rag": _run_rag,
        "chat": _run_chat,
    }
    results: dict[str, SystemResult] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {pool.submit(fn, query): name for name, fn in runners.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = SystemResult(answer="", elapsed=0, error=str(exc))
    return results


# ── Flatten question bank ────────────────────────────────────────────────────

def _flat_questions() -> list[dict[str, str]]:
    flat = []
    for category, qs in QUESTIONS.items():
        for q in qs:
            flat.append({"id": q["id"], "category": category, "question": q["q"]})
    return flat


# ── Incremental save helpers ─────────────────────────────────────────────────

def _save_json(
    all_results: list[dict[str, Any]],
    errors: int,
    total: int,
) -> Path:
    """Write JSON after every question so no progress is lost.

    Output location is ``benchmark/datasets/`` (not ``results/``) because the
    Preference page reads this file at runtime, and ``datasets/`` is the only
    directory that gets shipped into the production container
    (``benchmark/results/`` is excluded by ``.dockerignore``).
    """
    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": total,
        "total_answers": len(all_results) * len(SYSTEMS),
        "completed": len(all_results),
        "errors": errors,
        "display_names": DISPLAY_NAMES,
        "questions": all_results,
    }
    path = _DATASETS_DIR / "pregenerated_answers.json"
    path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _save_markdown(
    all_results: list[dict[str, Any]],
    errors: int,
    total: int,
) -> Path:
    """Write Markdown after every question so no progress is lost."""
    done = len(all_results)
    md_lines = [
        "# Câu trả lời từ 3 hệ thống",
        "",
        f"> Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Tiến độ: {done}/{total} câu | Tổng câu trả lời: {done * len(SYSTEMS)} | Lỗi: {errors}",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(all_results, 1):
        cat_short = entry["category"].split("—")[-1].strip() if "—" in entry["category"] else entry["category"]
        md_lines.append(f"## Câu {i}. {entry['question']}")
        md_lines.append("")
        md_lines.append(f"*{entry['id']} — {cat_short}*")
        md_lines.append("")

        for sys_key in DISPLAY_ORDER:
            display_name = DISPLAY_NAMES[sys_key]
            answer_data = entry["answers"].get(sys_key, {})
            answer = answer_data.get("answer", "")
            md_lines.append(f"### {display_name}")
            md_lines.append("")
            md_lines.append(answer if answer else "*(Lỗi — không có câu trả lời)*")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # System name mapping at the end (for researcher only)
    md_lines.append("## Bảng ánh xạ tên hệ thống (CHỈ DÀNH CHO NGHIÊN CỨU VIÊN)")
    md_lines.append("")
    md_lines.append("| Tên hiển thị | Hệ thống thực |")
    md_lines.append("|-------------|---------------|")
    for sys_key in DISPLAY_ORDER:
        md_lines.append(f"| {DISPLAY_NAMES[sys_key]} | {sys_key} |")
    md_lines.append("")

    path = _RESULTS_DIR / "pregenerated_answers.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def _load_existing() -> dict[str, dict[str, Any]]:
    """Load existing entries from pregenerated_answers.json as {id: entry}."""
    path = _DATASETS_DIR / "pregenerated_answers.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data.get("questions", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated CASE IDs to regenerate (e.g. CASE_063,CASE_078). "
             "Others are preserved from the existing JSON. "
             "Empty string means regenerate all.",
    )
    args = parser.parse_args()
    requested_ids = {x.strip() for x in args.ids.split(",") if x.strip()}

    all_questions = _flat_questions()
    if requested_ids:
        # Validate
        known = {q["id"] for q in all_questions}
        unknown = requested_ids - known
        if unknown:
            raise SystemExit(f"Unknown CASE IDs: {sorted(unknown)}")
        to_run = [q for q in all_questions if q["id"] in requested_ids]
        existing = _load_existing()
    else:
        to_run = all_questions
        existing = {}

    total_target = len(to_run)
    total_reported = len(all_questions) if requested_ids else total_target

    print(f"=== Batch Generate Answers ===")
    if requested_ids:
        print(f"Mode:      selective regeneration ({sorted(requested_ids)})")
        print(f"Preserved: {len(existing) - total_target} existing entries")
    else:
        print(f"Mode:      full run")
    print(f"To run:    {total_target}/{total_reported} questions")
    print(f"Systems:   {', '.join(SYSTEMS)}")
    print(f"Total answers to (re)generate: {total_target * len(SYSTEMS)}")
    print()

    # Seed results with preserved entries (kept in original order of QUESTIONS)
    all_results: list[dict[str, Any]] = []
    if requested_ids:
        for q in all_questions:
            if q["id"] not in requested_ids and q["id"] in existing:
                all_results.append(existing[q["id"]])

    errors = 0

    for idx, q in enumerate(to_run, 1):
        qid = q["id"]
        question = q["question"]
        category = q["category"]

        print(f"[{idx}/{total_target}] {qid}: {question[:60]}...", flush=True)
        t0 = time.time()
        results = _run_all(question)
        elapsed = time.time() - t0

        # Build answer entry
        answers_raw = {}
        for sys_name in SYSTEMS:
            r = results.get(sys_name, SystemResult(answer="", elapsed=0, error="not run"))
            answers_raw[sys_name] = {
                "answer": r.answer,
                "elapsed": round(r.elapsed, 2),
                "error": r.error,
                "metadata": r.metadata,
            }
            if r.error:
                errors += 1

        new_entry = {
            "id": qid,
            "category": category,
            "question": question,
            "answers": answers_raw,
        }

        # Safety guard: if all 3 systems error out (likely env/config failure),
        # preserve existing entry instead of overwriting with empty errors.
        all_errored = all(answers_raw[s].get("error") for s in SYSTEMS)
        if all_errored and qid in existing:
            print(f"  ⚠ All systems errored — preserving existing entry for {qid}",
                  flush=True)
            new_entry = existing[qid]

        # Replace the preserved copy if selective mode
        if requested_ids and any(e["id"] == qid for e in all_results):
            all_results = [new_entry if e["id"] == qid else e for e in all_results]
        else:
            all_results.append(new_entry)

        # Restore original QUESTIONS order before saving
        order = {q["id"]: i for i, q in enumerate(all_questions)}
        all_results.sort(key=lambda e: order.get(e["id"], 999))

        # Progress
        err_str = ""
        for sys_name in SYSTEMS:
            r = results.get(sys_name)
            if r and r.error:
                err_str += f"  [ERROR] {sys_name}: {r.error[:80]}\n"
        status = f"  Done in {elapsed:.1f}s"
        if err_str:
            status += f"\n{err_str}"
        print(status, flush=True)

        # Save incrementally after each question
        json_path = _save_json(all_results, errors, total_reported)
        md_path = _save_markdown(all_results, errors, total_reported)
        print(f"  Saved ({idx}/{total_target})", flush=True)

    print(f"\nJSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print(f"\nDone! {total_target * len(SYSTEMS)} answers (re)generated ({errors} errors).")


if __name__ == "__main__":
    main()
