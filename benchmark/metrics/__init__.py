"""
Benchmark metrics for GraphRAGv2.

Groups
------
IR    — Precision@K, Recall@K, F1@K, MAP, MRR, NDCG@K
NLG   — BLEU, ROUGE-1/2/L, METEOR
Exact — ExactMatch, KeywordCoverage, EntityCoverage
RAGAS — Faithfulness, AnswerRelevance, ContextPrecision,
         ContextRecall, ContextRelevance  (disabled by default)
"""

from .base import MetricGroup, MetricResult, MetricBase
from .registry import MetricRegistry
from .ir_metrics import PrecisionAtK, RecallAtK, F1AtK, MAPMetric, MRRMetric, NDCGAtK
from .nlg_metrics import BLEUMetric, ROUGE1Metric, ROUGE2Metric, ROUGELMetric, METEORMetric
from .exact_metrics import ExactMatchMetric, KeywordCoverageMetric, EntityCoverageMetric
from .ragas_metrics import (
    FaithfulnessMetric, AnswerRelevanceMetric,
    ContextPrecisionMetric, ContextRecallMetric, ContextRelevanceMetric,
)

__all__ = [
    "MetricGroup", "MetricResult", "MetricBase", "MetricRegistry",
    "PrecisionAtK", "RecallAtK", "F1AtK", "MAPMetric", "MRRMetric", "NDCGAtK",
    "BLEUMetric", "ROUGE1Metric", "ROUGE2Metric", "ROUGELMetric", "METEORMetric",
    "ExactMatchMetric", "KeywordCoverageMetric", "EntityCoverageMetric",
    "FaithfulnessMetric", "AnswerRelevanceMetric",
    "ContextPrecisionMetric", "ContextRecallMetric", "ContextRelevanceMetric",
]
