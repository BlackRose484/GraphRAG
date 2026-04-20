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
from .score_aggregator import aggregate as aggregate_scores

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
    category:   str = ""     # e.g. "local_queries" | "community_queries" | "global_queries"
    scores:          Dict[str, Optional[float]] = field(default_factory=dict)
    latency_s:       float = 0.0
    error:           Optional[str] = None
    retrieval_detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results for one pipeline over N questions."""
    pipeline:   str
    n_cases:    int
    cases:      List[CaseResult]                              = field(default_factory=list)
    averages:   Dict[str, Optional[float]]                    = field(default_factory=dict)
    by_category: Dict[str, Dict[str, Optional[float]]]        = field(default_factory=dict)
    metadata:   Dict[str, Any]                                = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "pipeline":    self.pipeline,
            "n_cases":     self.n_cases,
            "averages":    self.averages,
            "by_category": self.by_category,
            "cases":       [
                {
                    "case_id":   c.case_id,
                    "question":  c.question,
                    "category":  c.category,
                    "answer":    c.answer,
                    "reference": c.reference,
                    "scores":    c.scores,
                    "latency_s": c.latency_s,
                    "error":     c.error,
                }
                for c in self.cases
            ],
            "metadata":    self.metadata,
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
        case_ids:          Optional[List[str]] = None,
        progress_cb:       Optional[ProgressCB] = None,
        output_dir:        Optional[str | Path] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate one or both pipelines against a subset of the dataset.

        Args:
            dataset_path:     Path to CheoBench JSON file.
            graphrag_pipeline: ``GraphRAGPipeline`` instance (or ``None`` to skip).
            rag_pipeline:      ``VectorRAGPipeline`` instance (or ``None`` to skip).
            n_cases:           Max questions to evaluate (used only when
                ``case_ids`` is None).
            case_ids:          Explicit list of case IDs to evaluate. When
                provided, ``n_cases`` is ignored and only matching cases run.
            progress_cb:       ``(current, total, message)`` callback for UI.
            output_dir:        When set, results are persisted under this
                directory: ``meta.json`` (config), ``partial.jsonl`` (one
                line per case appended live, so a crash mid-run preserves
                everything completed so far), and ``final.json`` written
                only after all cases finish.

        Returns:
            List of :class:`BenchmarkResult` (one per active pipeline).
        """
        cases = self._load_dataset(dataset_path, n_cases, case_ids=case_ids)
        total = len(cases) * sum(
            [graphrag_pipeline is not None, rag_pipeline is not None]
        )

        out_path: Optional[Path] = None
        partial_fp = None
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            self._write_meta(
                out_path,
                dataset_path=dataset_path,
                pipelines=[n for n, p in
                           [("GraphRAG", graphrag_pipeline), ("RAG", rag_pipeline)]
                           if p is not None],
                case_ids=[c["id"] for c in cases],
                metric_names=[m.name for m in self.registry.active_metrics()],
            )
            partial_fp = open(out_path / "partial.jsonl", "w", encoding="utf-8")

        step  = 0
        results: List[BenchmarkResult] = []

        try:
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
                    logger.info("%s | %s → latency=%.2fs",
                                pipe_name, case["id"], result.latency_s)

                    # Stream per-case result to disk immediately
                    if partial_fp is not None:
                        partial_fp.write(json.dumps(
                            self._case_to_dict(result),
                            ensure_ascii=False,
                        ) + "\n")
                        partial_fp.flush()

                results.append(BenchmarkResult(
                    pipeline=pipe_name,
                    n_cases=len(pipe_cases),
                    cases=pipe_cases,
                    averages=self._aggregate(pipe_cases),
                    by_category=self._aggregate_by_category(pipe_cases),
                ))
        finally:
            if partial_fp is not None:
                partial_fp.close()

        # Write final aggregated file once all pipelines finish
        if out_path is not None:
            (out_path / "final.json").write_text(
                json.dumps(
                    {
                        "results":   [r.to_dict() for r in results],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )

        return results

    @staticmethod
    def _write_meta(
        out_path:     Path,
        dataset_path: str | Path,
        pipelines:    List[str],
        case_ids:     List[str],
        metric_names: List[str],
    ) -> None:
        from src.core.settings import settings
        meta = {
            "started_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset":      str(dataset_path),
            "llm_model":    settings.llm.model,
            "pipelines":    pipelines,
            "n_cases":      len(case_ids),
            "case_ids":     case_ids,
            "metrics":      metric_names,
        }
        (out_path / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _case_to_dict(c: "CaseResult") -> Dict[str, Any]:
        """Serialise a single CaseResult for the streaming partial.jsonl file."""
        return {
            "case_id":   c.case_id,
            "pipeline":  c.pipeline,
            "category":  c.category,
            "question":  c.question,
            "answer":    c.answer,
            "reference": c.reference,
            "scores":    c.scores,
            "latency_s": c.latency_s,
            "error":     c.error,
        }

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
            if mr.error:
                logger.warning(
                    "Metric %s failed for %s/%s: %s",
                    metric.name, pipe_name, case["id"], mr.error,
                )

        # Composite weighted scores from score_aggregator
        composites = aggregate_scores(scores).as_dict()
        scores.update(composites)

        return CaseResult(
            case_id=case["id"],
            question=question,
            pipeline=pipe_name,
            answer=answer,
            reference=reference,
            category=case.get("category", ""),
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

    @classmethod
    def _aggregate_by_category(
        cls, cases: List[CaseResult]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Group cases by ``category`` and aggregate each group separately.

        Lets the benchmark distinguish "RAG wins on simple lookup" from
        "GraphRAG wins on multi-hop" — which a single global average hides.
        """
        if not cases:
            return {}
        buckets: Dict[str, List[CaseResult]] = {}
        for c in cases:
            buckets.setdefault(c.category or "uncategorized", []).append(c)
        return {cat: cls._aggregate(group) for cat, group in buckets.items()}

    # ── Dataset loader ────────────────────────────────────────────────────────

    @staticmethod
    def _load_dataset(
        path: str | Path,
        n: int,
        case_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw_cases = data.get("test_cases", data) if isinstance(data, dict) else data

        normalized = [
            {
                "id":           c.get("id", f"CASE_{i+1:03d}"),
                "category":     c.get("category", ""),
                "question":     c.get("question", ""),
                "ground_truth": c.get("ground_truth", {}),
            }
            for i, c in enumerate(raw_cases)
            if c.get("question")
        ]

        if case_ids:
            # Preserve user-specified order; silently skip unknown IDs.
            by_id = {c["id"]: c for c in normalized}
            return [by_id[cid] for cid in case_ids if cid in by_id]

        return normalized[:n]

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
