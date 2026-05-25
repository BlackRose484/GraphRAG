"""MetricRegistry — enable / disable metrics and groups at runtime."""

from __future__ import annotations

from typing import Dict, List

from .base import MetricBase, MetricGroup


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics:  Dict[str, MetricBase] = {}
        self._enabled:  Dict[str, bool]       = {}

    def register(self, metric: MetricBase, enabled: bool = True) -> "MetricRegistry":
        self._metrics[metric.name] = metric
        self._enabled[metric.name] = enabled
        return self

    def enable(self, name: str) -> None:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name!r}")
        self._enabled[name] = True

    def disable(self, name: str) -> None:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name!r}")
        self._enabled[name] = False

    def enable_group(self, group: MetricGroup) -> None:
        for name, m in self._metrics.items():
            if m.group == group:
                self._enabled[name] = True

    def disable_group(self, group: MetricGroup) -> None:
        for name, m in self._metrics.items():
            if m.group == group:
                self._enabled[name] = False

    def set_enabled(self, name: str, value: bool) -> None:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name!r}")
        self._enabled[name] = value

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, False)

    def active_metrics(self) -> List[MetricBase]:
        return [m for name, m in self._metrics.items() if self._enabled[name]]

    def all_metrics(self) -> List[MetricBase]:
        return list(self._metrics.values())

    def all_names(self) -> List[str]:
        return list(self._metrics.keys())

    def by_group(self, group: MetricGroup) -> List[MetricBase]:
        return [m for m in self._metrics.values() if m.group == group]

    @classmethod
    def default(cls) -> "MetricRegistry":
        from .ir_metrics import PrecisionMetric, RecallMetric, MAPMetric, NDCGAtK
        from .exact_metrics import ExactMatchMetric, KeywordCoverageMetric, EntityCoverageMetric
        from .ragas_metrics import (
            FaithfulnessMetric, AnswerRelevanceMetric,
            ContextPrecisionMetric, ContextRecallMetric, ContextRelevanceMetric,
            ContextEntitiesRecallMetric,
        )

        registry = cls()

        for m in [PrecisionMetric(), RecallMetric(), MAPMetric(), NDCGAtK()]:
            registry.register(m, enabled=True)

        # KeywordCoverage / EntityCoverage rely on literal substring match,
        # which under-scores semantically correct paraphrases. Faithfulness
        # and AnswerRelevance (RAGAs) cover the same intent with better
        # semantic understanding. ExactMatch is always ≈0 for generative QA.
        registry.register(ExactMatchMetric(),       enabled=False)
        registry.register(KeywordCoverageMetric(),  enabled=False)
        registry.register(EntityCoverageMetric(),   enabled=False)

        # ContextRelevance overlaps with ContextPrecision and isn't in the
        # weight table — registered but disabled.
        for m in [FaithfulnessMetric(), AnswerRelevanceMetric(),
                  ContextPrecisionMetric(), ContextRecallMetric(),
                  ContextEntitiesRecallMetric()]:
            registry.register(m, enabled=True)
        registry.register(ContextRelevanceMetric(), enabled=False)

        return registry
