"""
Exact-match and keyword/entity coverage metrics.

These are fast, reference-based metrics that require no external libraries
and no LLM calls.
"""

from __future__ import annotations

import re
from typing import List, Set

from .base import MetricBase, MetricGroup


def _normalize(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics are kept (important for Chèo names)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains(haystack: str, needle: str) -> bool:
    """Case-insensitive substring check."""
    return _normalize(needle) in _normalize(haystack)


# ── Exact Match ───────────────────────────────────────────────────────────────

class ExactMatchMetric(MetricBase):
    """
    Binary exact match after normalization (lowercase, collapse whitespace).

    Returns 1.0 if the normalized hypothesis equals the normalized reference,
    otherwise 0.0.
    """

    @property
    def name(self) -> str:         return "ExactMatch"
    @property
    def group(self) -> MetricGroup: return MetricGroup.EXACT
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, reference: str, **_) -> float:
        """
        Args:
            hypothesis: Model-generated answer.
            reference:  Ground-truth answer string.
        """
        if not hypothesis or not reference:
            return 0.0
        return 1.0 if _normalize(hypothesis) == _normalize(reference) else 0.0


# ── Keyword Coverage ──────────────────────────────────────────────────────────

class KeywordCoverageMetric(MetricBase):
    """
    Fraction of ``must_include_keywords`` from the dataset ground truth that
    appear (as substrings) in the generated answer.

        KeywordCoverage = |found keywords| / |total keywords|
    """

    @property
    def name(self) -> str:         return "KeywordCoverage"
    @property
    def group(self) -> MetricGroup: return MetricGroup.EXACT
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, keywords: List[str], **_) -> float:
        """
        Args:
            hypothesis: Model-generated answer.
            keywords:   List of required keywords from ground-truth
                        (``ground_truth.must_include_keywords`` in CheoBench).
        """
        if not keywords:
            return 1.0   # no keywords required — trivially covered
        if not hypothesis:
            return 0.0
        found = sum(1 for kw in keywords if _contains(hypothesis, kw))
        return found / len(keywords)


# ── Entity Coverage ───────────────────────────────────────────────────────────

class EntityCoverageMetric(MetricBase):
    """
    Fraction of ``related_entities`` from the dataset ground truth that are
    mentioned in the generated answer.

        EntityCoverage = |mentioned entities| / |total entities|
    """

    @property
    def name(self) -> str:         return "EntityCoverage"
    @property
    def group(self) -> MetricGroup: return MetricGroup.EXACT
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, entities: List[str], **_) -> float:
        """
        Args:
            hypothesis: Model-generated answer.
            entities:   List of related entity names from ground-truth
                        (``ground_truth.related_entities`` in CheoBench).
        """
        if not entities:
            return 1.0
        if not hypothesis:
            return 0.0
        found = sum(1 for ent in entities if _contains(hypothesis, ent))
        return found / len(entities)
