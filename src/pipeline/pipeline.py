"""GraphRAG pipeline — top-level orchestrator."""
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

    When ``auto_routing=True`` and a slot is ``None``, the LLM-classified
    query_type drives retrieval methods and generation strategy:
        Local     → [nodes, triplets]          + Pre-Generation
        Community → [nodes, triplets, paths]   + Mid-Generation
        Global    → [paths, subgraph]          + Post-Generation
    Explicit args override auto-routing for that slot.
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

    @property
    def strategy(self) -> str:
        return self._generation.strategy_name

    def switch_strategy(self, strategy: str) -> None:
        """Hot-swap the generation strategy without creating a new pipeline."""
        self._generation.switch_strategy(strategy)
        self._user_strategy = strategy  # pin: subsequent runs respect this

    def run(self, query: str) -> PipelineResult:
        """Execute the full retrieval → generation pipeline."""
        start = time.perf_counter()

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

        # Auto-route generation strategy only if user didn't pin one
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
