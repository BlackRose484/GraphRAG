"""Exact-match and keyword/entity coverage metrics."""

from __future__ import annotations

import re
from typing import List

from .base import MetricBase, MetricGroup


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace; diacritics preserved for Chèo names."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains(haystack: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(haystack)


class ExactMatchMetric(MetricBase):
    @property
    def name(self) -> str:         return "ExactMatch"
    @property
    def group(self) -> MetricGroup: return MetricGroup.EXACT
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, reference: str, **_) -> float:
        if not hypothesis or not reference:
            return 0.0
        return 1.0 if _normalize(hypothesis) == _normalize(reference) else 0.0


class KeywordCoverageMetric(MetricBase):
    """KeywordCoverage = |found keywords| / |total keywords|."""

    @property
    def name(self) -> str:         return "KeywordCoverage"
    @property
    def group(self) -> MetricGroup: return MetricGroup.EXACT
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, keywords: List[str], **_) -> float:
        if not keywords:
            return 1.0
        if not hypothesis:
            return 0.0
        found = sum(1 for kw in keywords if _contains(hypothesis, kw))
        return found / len(keywords)


class EntityCoverageMetric(MetricBase):
    """EntityCoverage = |mentioned entities| / |total entities|."""

    @property
    def name(self) -> str:         return "EntityCoverage"
    @property
    def group(self) -> MetricGroup: return MetricGroup.EXACT
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, entities: List[str], **_) -> float:
        if not entities:
            return 1.0
        if not hypothesis:
            return 0.0
        found = sum(1 for ent in entities if _contains(hypothesis, ent))
        return found / len(entities)
