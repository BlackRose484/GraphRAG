"""
Benchmark metrics for GraphRAGv2.

Groups
------
IR    — Precision, Recall, MAP, MRR, NDCG@K
Exact — ExactMatch, KeywordCoverage, EntityCoverage
RAGAS — Faithfulness, AnswerRelevance, ContextPrecision,
         ContextRecall, ContextRelevance  (disabled by default)
"""

from .base import MetricGroup, MetricResult, MetricBase
from .registry import MetricRegistry
from .ir_metrics import PrecisionMetric, RecallMetric, MAPMetric, MRRMetric, NDCGAtK
from .exact_metrics import ExactMatchMetric, KeywordCoverageMetric, EntityCoverageMetric
from .ragas_metrics import (
    FaithfulnessMetric, AnswerRelevanceMetric,
    ContextPrecisionMetric, ContextRecallMetric, ContextRelevanceMetric,
    ContextEntitiesRecallMetric,
)

__all__ = [
    "MetricGroup", "MetricResult", "MetricBase", "MetricRegistry",
    "PrecisionMetric", "RecallMetric", "MAPMetric", "MRRMetric", "NDCGAtK",
    "ExactMatchMetric", "KeywordCoverageMetric", "EntityCoverageMetric",
    "FaithfulnessMetric", "AnswerRelevanceMetric",
    "ContextPrecisionMetric", "ContextRecallMetric", "ContextRelevanceMetric",
    "ContextEntitiesRecallMetric",
]
