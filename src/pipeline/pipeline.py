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

    Args:
        client: Connected :class:`~src.graph_loader.neo4j_client.Neo4jClient`.
        retrieval_methods: Subset of ``RetrievalMethod.ALL``.
        format_keys: Subset of :class:`~src.constants.constant.FormatKey` values.
        generation_strategy: One of ``'pre'``, ``'mid'``, ``'post'``.
        enable_query_enhancement: Expand/decompose the query before retrieval.
    """

    def __init__(
        self,
        client: Neo4jClient,
        *,
        retrieval_methods: list[str] | None = None,
        format_keys: list[str] | None = None,
        generation_strategy: str = GenerationStrategy.DEFAULT,
        enable_query_enhancement: bool = True,
    ) -> None:
        self._retrieval_methods = retrieval_methods or _DEFAULT_RETRIEVAL_METHODS
        self._format_keys = format_keys or _DEFAULT_FORMATS
        self._enable_enhancement = enable_query_enhancement

        self._retrieval = RetrievalOrchestrator(client)
        self._generation = GenerationOrchestrator(generation_strategy)

        _logger.info(
            "GraphRAGPipeline ready — strategy=%s  methods=%s  formats=%s",
            generation_strategy,
            self._retrieval_methods,
            self._format_keys,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def strategy(self) -> str:
        return self._generation.strategy_name

    def switch_strategy(self, strategy: str) -> None:
        """Hot-swap the generation strategy without creating a new pipeline."""
        self._generation.switch_strategy(strategy)

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
            retrieval_methods=self._retrieval_methods,
            format_keys=self._format_keys,
            enable_enhancement=self._enable_enhancement,
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

        # ── Stage 2: Generation ─────────────────────────────────────────────
        _logger.info("Pipeline stage 2: generation  strategy=%s", self.strategy)
        generation_result = self._generation.generate(
            query=query,
            graph_data=retrieval_result.graph_data,
            selected_formats=self._format_keys,
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
