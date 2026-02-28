"""
G-Generation orchestrator.

Wraps the three generation strategies (Pre / Mid / Post) and produces a
:class:`GenerationResult` dataclass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.constants.constant import FormatKey, GenerationStrategy
from src.core.settings import settings
from src.g_generation.strategies import STRATEGY_REGISTRY, BaseGenerationStrategy
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_DEFAULT_FORMATS: list[str] = FormatKey.DEFAULT_MID

GraphData = dict[str, Any]


@dataclass
class GenerationResult:
    """Output from one generation pipeline run."""

    query: str
    response: str
    strategy: str
    generation_time: float
    context_length: int     # len(context) passed to the model
    key_facts_used: str
    model_name: str
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "model": self.model_name,
            "generation_time_s": round(self.generation_time, 3),
            "context_length": self.context_length,
            "error": self.error,
        }


class GenerationOrchestrator:
    """Manage generation using a configurable strategy.

    Args:
        strategy: One of ``'pre'``, ``'mid'`` (default), ``'post'``.

    Raises:
        ValueError: If an unknown strategy name is provided.
    """

    def __init__(self, strategy: str = GenerationStrategy.DEFAULT) -> None:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid values: {sorted(STRATEGY_REGISTRY)}"
            )
        self._strategy_name = strategy
        self._strategy: BaseGenerationStrategy = STRATEGY_REGISTRY[strategy]()
        _logger.info("GenerationOrchestrator initialised with strategy='%s'", strategy)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    def generate(
        self,
        query: str,
        graph_data: GraphData,
        *,
        selected_formats: list[str] | None = None,
        key_facts: str = "",
    ) -> GenerationResult:
        """Run the configured strategy and return a :class:`GenerationResult`.

        Args:
            query: User query.
            graph_data: Retrieved graph data from :class:`~src.g_retrieval.orchestrator.RetrievalOrchestrator`.
            selected_formats: :class:`~src.constants.constant.FormatKey` values to include.
            key_facts: Pre-computed key facts string (computed fresh if empty).

        Returns:
            :class:`GenerationResult` — *error* is set on failure, response is empty string.
        """
        formats = selected_formats or _DEFAULT_FORMATS
        start = time.perf_counter()

        try:
            _logger.info("Generation start  strategy=%s  query=%r", self._strategy_name, query[:60])

            # Build context length estimate for the result metadata
            # (We re-use the same _build_context the strategy will call)
            ctx_preview = self._strategy._build_context(graph_data, formats)
            context_length = len(ctx_preview)

            response = self._strategy.generate(
                query=query,
                graph_data=graph_data,
                selected_formats=formats,
                key_facts=key_facts,
            )
            elapsed = time.perf_counter() - start
            result = GenerationResult(
                query=query,
                response=response,
                strategy=self._strategy_name,
                generation_time=elapsed,
                context_length=context_length,
                key_facts_used=key_facts,
                model_name=settings.llm.model,
            )
            _logger.info("Generation done: %s", result.summary())
            return result

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            _logger.error("Generation failed: %s", exc, exc_info=True)
            return GenerationResult(
                query=query,
                response="",
                strategy=self._strategy_name,
                generation_time=elapsed,
                context_length=0,
                key_facts_used=key_facts,
                model_name=settings.llm.model,
                error=str(exc),
            )

    def switch_strategy(self, strategy: str) -> None:
        """Hot-swap the strategy without recreating the orchestrator.

        Args:
            strategy: New strategy name (``'pre'``, ``'mid'``, or ``'post'``).
        """
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{strategy}'")
        self._strategy_name = strategy
        self._strategy = STRATEGY_REGISTRY[strategy]()
        _logger.info("Strategy switched to '%s'", strategy)
