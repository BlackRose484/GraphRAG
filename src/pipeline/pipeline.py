"""
GraphRAG pipeline — top-level orchestrator.

Sequences: RetrievalOrchestrator → GenerationOrchestrator and exposes a single
``run(query)`` entry-point used by both the Streamlit UI and the benchmark runner.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.constants.constant import (
    FormatKey,
    GenerationStrategy,
    QueryType,
    RetrievalMethod,
)
from src.core.settings import settings
from src.g_generation.orchestrator import GenerationOrchestrator, GenerationResult
from src.g_retrieval.orchestrator import RetrievalOrchestrator, RetrievalResult
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_DEFAULT_RETRIEVAL_METHODS = RetrievalMethod.DEFAULT
_DEFAULT_FORMATS = FormatKey.DEFAULT_MID


@dataclass
class PipelineResult:
    """Combined output from one pipeline run."""

    query: str
    retrieval: RetrievalResult
    generation: GenerationResult
    total_time: float
    success: bool

    @property
    def answer(self) -> str:
        """Shortcut to the final generated answer."""
        return self.generation.response

    def summary(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "success": self.success,
            "total_time_s": round(self.total_time, 3),
            "retrieval": self.retrieval.summary(),
            "generation": self.generation.summary(),
        }


class GraphRAGPipeline:
    """Full GraphRAG pipeline: retrieval + generation.

    Auto-routing
    ------------
    When ``auto_routing=True`` (default) and the corresponding parameter is
    left as ``None``, the pipeline lets the LLM-classified ``query_type``
    drive both the retrieval method set and the generation strategy:

        Local     → [nodes, triplets]                 + Pre-Generation
        Community → [nodes, triplets, paths]          + Mid-Generation
        Global    → [nodes, triplets, paths, subgraph] + Post-Generation

    Any explicitly-passed argument overrides auto-routing for that slot —
    benchmark runs use this to measure each configuration in isolation.

    Args:
        client: Connected :class:`~src.graph_loader.neo4j_client.Neo4jClient`.
        retrieval_methods: Subset of ``RetrievalMethod.ALL``; ``None`` = let
                           auto-routing decide per query.
        format_keys: Subset of :class:`~src.constants.constant.FormatKey` values;
                     ``None`` = use the active strategy's default formats.
        generation_strategy: ``'pre'``/``'mid'``/``'post'`` or ``None`` for
                             auto-routing per query.
        enable_query_enhancement: Expand/decompose the query before retrieval.
                                  Set to ``False`` ONLY for user-study fair
                                  comparison against vector RAG (which has no
                                  query rewriting). See orchestrator module
                                  docstring for the BASIC vs ENHANCED paths.
        auto_routing: Use LLM query classification to pick methods/strategy.
    """

    def __init__(
        self,
        client: Neo4jClient,
        *,
        retrieval_methods: list[str] | None = None,
        format_keys: list[str] | None = None,
        generation_strategy: str | None = None,
        enable_query_enhancement: bool = True,
        auto_routing: bool = True,
    ) -> None:
        # Distinguish "user pinned" from "let auto-routing decide".
        self._user_methods   = retrieval_methods
        self._user_formats   = format_keys
        self._user_strategy  = generation_strategy
        self._enable_enhancement = enable_query_enhancement
        self._auto_routing   = auto_routing

        self._retrieval  = RetrievalOrchestrator(client)
        self._generation = GenerationOrchestrator(
            generation_strategy or GenerationStrategy.DEFAULT
        )

        _logger.info(
            "GraphRAGPipeline ready — auto_routing=%s  strategy=%s  methods=%s  formats=%s",
            auto_routing,
            generation_strategy or "(auto)",
            retrieval_methods or "(auto)",
            format_keys or "(strategy default)",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def strategy(self) -> str:
        return self._generation.strategy_name

    def switch_strategy(self, strategy: str) -> None:
        """Hot-swap the generation strategy without creating a new pipeline."""
        self._generation.switch_strategy(strategy)
        self._user_strategy = strategy  # pin: subsequent runs respect this

    def run(self, query: str) -> PipelineResult:
        """Execute the full retrieval → generation pipeline.

        Args:
            query: Raw user question in Vietnamese.

        Returns:
            :class:`PipelineResult`.  On failure *success* is ``False`` and
            *answer* is an empty string.
        """
        start = time.perf_counter()

        # ── Stage 1: Retrieval ──────────────────────────────────────────────
        _logger.info("Pipeline stage 1: retrieval  query=%r", query[:60])
        retrieval_result = self._retrieval.retrieve(
            query,
            retrieval_methods=self._user_methods,
            format_keys=self._user_formats or _DEFAULT_FORMATS,
            enable_enhancement=self._enable_enhancement,
            auto_routing=self._auto_routing,
        )

        if retrieval_result.error:
            _logger.error("Retrieval error: %s", retrieval_result.error)
            elapsed = time.perf_counter() - start
            failed_gen = GenerationResult(
                query=query,
                response="",
                strategy=self._generation.strategy_name,
                generation_time=0.0,
                context_length=0,
                key_facts_used="",
                model_name=settings.llm.model,
                error=f"skipped — retrieval failed: {retrieval_result.error}",
            )
            return PipelineResult(
                query=query,
                retrieval=retrieval_result,
                generation=failed_gen,
                total_time=elapsed,
                success=False,
            )

        # ── Auto-route generation strategy (only if user didn't pin one) ────
        if (
            self._auto_routing
            and self._user_strategy is None
            and retrieval_result.query_type
        ):
            target_strategy = QueryType.STRATEGY.get(
                retrieval_result.query_type, GenerationStrategy.DEFAULT
            )
            if target_strategy != self._generation.strategy_name:
                _logger.info(
                    "Auto-routing: query_type=%s → strategy=%s",
                    retrieval_result.query_type, target_strategy,
                )
                self._generation.switch_strategy(target_strategy)

        # Pick formats: explicit user choice > strategy's default > pipeline default
        formats = self._user_formats or self._generation.default_formats

        # ── Stage 2: Generation ─────────────────────────────────────────────
        _logger.info("Pipeline stage 2: generation  strategy=%s", self.strategy)
        generation_result = self._generation.generate(
            query=query,
            graph_data=retrieval_result.graph_data,
            selected_formats=formats,
            key_facts=retrieval_result.key_facts,
        )

        elapsed = time.perf_counter() - start
        success = generation_result.error is None

        result = PipelineResult(
            query=query,
            retrieval=retrieval_result,
            generation=generation_result,
            total_time=elapsed,
            success=success,
        )
        _logger.info("Pipeline complete: %s", result.summary())
        return result
