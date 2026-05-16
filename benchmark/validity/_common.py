"""
Shared paths, loaders, and helpers for the ``benchmark/validity/*`` scripts.

Three sibling scripts (``dataset_quality``, ``intrinsic``, ``convergent``) all
consume the same CheoBench dataset and the same baseline benchmark run, so
those locations live here as single sources of truth.

Override the benchmark run from the shell::

    BENCHMARK_RUN_DIR=2026-05-01_12-00-00 python -m benchmark.validity.intrinsic
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
CHEOBENCH = ROOT / "benchmark" / "datasets" / "CheoBench_v2.json"

# Benchmark run all validity scripts compare against. Override via env var.
_DEFAULT_RUN_DIR = "2026-04-19_11-48-08"
BENCHMARK_RUN = (
    ROOT
    / "benchmark"
    / "results"
    / "auto_benchmark"
    / os.getenv("BENCHMARK_RUN_DIR", _DEFAULT_RUN_DIR)
)
FINAL_PATH = BENCHMARK_RUN / "final.json"
PARTIAL_PATH = BENCHMARK_RUN / "partial.jsonl"

# Output directory (gitignored — regenerated from data on each run)
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


# ── Question categories ───────────────────────────────────────────────────────

CATEGORY_ORDER: list[str] = ["local_queries", "community_queries", "global_queries"]

CATEGORY_LABEL: dict[str, str] = {
    "local_queries":     "Local",
    "community_queries": "Community",
    "global_queries":    "Global",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_cheobench() -> list[dict[str, Any]]:
    """Return the ``test_cases`` list from the CheoBench dataset."""
    return json.loads(CHEOBENCH.read_text(encoding="utf-8"))["test_cases"]


def load_final() -> dict[str, Any]:
    """Return the full ``final.json`` of the baseline benchmark run."""
    return json.loads(FINAL_PATH.read_text(encoding="utf-8"))


def load_partial() -> list[dict[str, Any]]:
    """Return per-case records streamed to ``partial.jsonl``."""
    out: list[dict[str, Any]] = []
    with PARTIAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_per_case_scores() -> dict[str, dict[str, float]]:
    """Return ``{case_id: {pipeline: S_overall}}`` from the baseline run."""
    data = load_final()
    out: dict[str, dict[str, float]] = {}
    for r in data["results"]:
        pipeline = r["pipeline"]
        for c in r["cases"]:
            out.setdefault(c["case_id"], {})[pipeline] = c["scores"]["S_overall"]
    return out


# ── Statistical helpers ───────────────────────────────────────────────────────

def sig(p: float) -> str:
    """Return ``***`` / ``**`` / ``*`` / ``ns`` significance marker for *p*.

    Returns ``n/a`` when *p* is NaN or not a number.
    """
    try:
        if math.isnan(float(p)):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
