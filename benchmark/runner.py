"""
BenchmarkRunner — orchestrates evaluation of GraphRAG and/or RAG pipelines
against the CheoBench dataset.

Usage
-----
    from benchmark.runner import BenchmarkRunner
    from benchmark.metrics import MetricRegistry

    registry = MetricRegistry.default()
    registry.disable_group(MetricGroup.RAGAS)

    runner  = BenchmarkRunner(registry=registry)
    results = runner.run(
        dataset_path="benchmark/datasets/CheoBench_v2.json",
        graphrag_pipeline=graphrag_pipe,   # or None
        rag_pipeline=rag_pipe,             # or None
        n_cases=20,
        progress_cb=lambda i, total, msg: print(f"{i}/{total} {msg}"),
    )
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .metrics.base import MetricGroup, MetricResult
from .metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    """Metric scores for a single benchmark question."""
    case_id:    str
    question:   str
    pipeline:   str          # "graphrag" | "rag"
    answer:     str
    reference:  str
    scores:          Dict[str, Optional[float]] = field(default_factory=dict)
    latency_s:       float = 0.0
    error:           Optional[str] = None
    retrieval_detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results for one pipeline over N questions."""
    pipeline:   str
    n_cases:    int
    cases:      List[CaseResult]                    = field(default_factory=list)
    averages:   Dict[str, Optional[float]]          = field(default_factory=dict)
    metadata:   Dict[str, Any]                      = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "pipeline":  self.pipeline,
            "n_cases":   self.n_cases,
            "averages":  self.averages,
            "cases":     [
                {
                    "case_id":   c.case_id,
                    "question":  c.question,
                    "answer":    c.answer,
                    "reference": c.reference,
                    "scores":    c.scores,
                    "latency_s": c.latency_s,
                    "error":     c.error,
                }
                for c in self.cases
            ],
            "metadata":  self.metadata,
        }


# ── Runner ────────────────────────────────────────────────────────────────────

ProgressCB = Callable[[int, int, str], None]


class BenchmarkRunner:
    """
    Runs benchmark evaluation against CheoBench JSON dataset.

    Args:
        registry: Which metrics to evaluate.  Defaults to
                  ``MetricRegistry.default()`` (RAGAS disabled).
    """

    def __init__(self, registry: Optional[MetricRegistry] = None) -> None:
        self.registry = registry or MetricRegistry.default()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        dataset_path:      str | Path,
        graphrag_pipeline  = None,
        rag_pipeline       = None,
        n_cases:           int = 20,
        progress_cb:       Optional[ProgressCB] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate one or both pipelines against the first *n_cases* questions.

        Args:
            dataset_path:     Path to CheoBench JSON file.
            graphrag_pipeline: ``GraphRAGPipeline`` instance (or ``None`` to skip).
            rag_pipeline:      ``VectorRAGPipeline`` instance (or ``None`` to skip).
            n_cases:           Max questions to evaluate.
            progress_cb:       ``(current, total, message)`` callback for UI.

        Returns:
            List of :class:`BenchmarkResult` (one per active pipeline).
        """
        cases = self._load_dataset(dataset_path, n_cases)
        total = len(cases) * sum(
            [graphrag_pipeline is not None, rag_pipeline is not None]
        )
        step  = 0
        results: List[BenchmarkResult] = []

        for pipe_name, pipeline in [("GraphRAG", graphrag_pipeline),
                                     ("RAG",      rag_pipeline)]:
            if pipeline is None:
                continue

            pipe_cases: List[CaseResult] = []
            for case in cases:
                step += 1
                if progress_cb:
                    progress_cb(step, total, f"[{pipe_name}] {case['id']}")

                result = self._run_case(pipe_name, case, pipeline)
                pipe_cases.append(result)
                logger.info("%s | %s → latency=%.2fs", pipe_name, case["id"], result.latency_s)

            results.append(BenchmarkResult(
                pipeline=pipe_name,
                n_cases=len(pipe_cases),
                cases=pipe_cases,
                averages=self._aggregate(pipe_cases),
            ))

        return results

    # ── Case runner ───────────────────────────────────────────────────────────

    def _run_case(self, pipe_name: str, case: Dict, pipeline) -> CaseResult:
        question  = case["question"]
        gt        = case.get("ground_truth", {})
        reference = gt.get("answer", "")
        keywords  = gt.get("must_include_keywords", [])
        entities  = gt.get("related_entities", [])

        t0 = time.time()
        answer = ""
        error  = None
        context = ""
        retrieved_names: List[str] = []

        retrieval_detail: Dict[str, Any] = {}
        try:
            result  = pipeline.run(question)
            answer  = result.answer or ""
            # Extract context string and retrieved entity names
            context = self._extract_context(result)
            retrieved_names = self._extract_entity_names(result)
            # Store full retrieval detail for UI inspection
            retrieval_detail = self._extract_retrieval_detail(result)
        except Exception as exc:
            error = str(exc)
            logger.warning("%s case %s failed: %s", pipe_name, case["id"], exc)

        latency = time.time() - t0

        # ── Compute all active metrics ────────────────────────────────────────
        scores: Dict[str, Optional[float]] = {}
        for metric in self.registry.active_metrics():
            mr = metric.safe_evaluate(
                # IR kwargs
                retrieved=retrieved_names,
                relevant=set(entities),
                # NLG kwargs
                hypothesis=answer,
                reference=reference,
                # Exact kwargs
                keywords=keywords,
                entities=entities,
                # RAGAS kwargs
                question=question,
                context=context,
            )
            scores[metric.name] = mr.value

        return CaseResult(
            case_id=case["id"],
            question=question,
            pipeline=pipe_name,
            answer=answer,
            reference=reference,
            scores=scores,
            latency_s=latency,
            error=error,
            retrieval_detail=retrieval_detail,
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(cases: List[CaseResult]) -> Dict[str, Optional[float]]:
        """Average scores across all cases, ignoring None values."""
        if not cases:
            return {}
        all_names = list(cases[0].scores.keys())
        averages: Dict[str, Optional[float]] = {}
        for name in all_names:
            vals = [c.scores[name] for c in cases if c.scores.get(name) is not None]
            averages[name] = sum(vals) / len(vals) if vals else None
        return averages

    # ── Dataset loader ────────────────────────────────────────────────────────

    @staticmethod
    def _load_dataset(path: str | Path, n: int) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cases = data.get("test_cases", data) if isinstance(data, dict) else data
        return [
            {
                "id":           c.get("id", f"CASE_{i+1:03d}"),
                "category":     c.get("category", ""),
                "question":     c.get("question", ""),
                "ground_truth": c.get("ground_truth", {}),
            }
            for i, c in enumerate(cases[:n])
            if c.get("question")
        ]

    # ── Context/entity extraction from pipeline results ───────────────────────

    @staticmethod
    def _extract_retrieval_detail(result) -> Dict[str, Any]:
        """Build the retrieval_detail dict stored per CaseResult for UI display."""
        detail: Dict[str, Any] = {}
        try:
            ret = result.retrieval
            detail["nodes"]             = list(ret.graph_data.get("nodes",    []))
            detail["triplets"]          = list(ret.graph_data.get("triplets", []))
            detail["paths"]             = list(ret.graph_data.get("paths",    []))
            detail["formatted_contexts"] = dict(ret.formatted_contexts or {})
            detail["processed_query"]   = dict(ret.processed_query or {})
            detail["num_nodes"]         = len(detail["nodes"])
            detail["num_triplets"]      = len(detail["triplets"])
            detail["num_paths"]         = len(detail["paths"])
        except AttributeError:
            pass
        return detail

    @staticmethod
    def _extract_context(result) -> str:
        """Pull context string from GraphRAG or RAG pipeline result."""
        try:
            fmts = result.retrieval.formatted_contexts
            if fmts:
                return "\n\n".join(fmts.values())
        except AttributeError:
            pass
        return ""

    @staticmethod
    def _extract_entity_names(result) -> List[str]:
        """Extract entity name strings from the retrieved graph_data nodes."""
        names: List[str] = []
        try:
            nodes = result.retrieval.graph_data.get("nodes", [])
            for n in nodes:
                name = n.get("name") or n.get("charName") or n.get("title") or ""
                if name:
                    names.append(name)
        except AttributeError:
            pass
        return names
