"""
Information Retrieval metrics.

References
----------
Manning, C. D., Raghavan, P., & Schütze, H. (2008).
    Introduction to Information Retrieval. Cambridge University Press.
Järvelin, K., & Kekäläinen, J. (2002).
    Cumulated gain-based evaluation of IR techniques. ACM TOIS, 20(4), 422–446.
"""

from __future__ import annotations

import math
from typing import List, Set

from .base import MetricBase, MetricGroup


# ── Shared helper ─────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.strip().lower()

def _is_hit(item: str, relevant: Set[str]) -> bool:
    """Case-insensitive exact or substring match."""
    item_n = _norm(item)
    return any(item_n == _norm(r) or _norm(r) in item_n or item_n in _norm(r)
               for r in relevant)


# ── Precision@K ───────────────────────────────────────────────────────────────

class PrecisionAtK(MetricBase):
    """Precision@K = |relevant ∩ retrieved[:K]| / K"""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    @property
    def name(self) -> str:         return f"Precision@{self.k}"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        """
        Args:
            retrieved: Ordered list of retrieved entity/node names.
            relevant:  Ground-truth set of relevant entity names.
        """
        if not retrieved:
            return 0.0
        top = retrieved[: self.k]
        hits = sum(1 for item in top if _is_hit(item, relevant))
        return hits / self.k


# ── Recall@K ──────────────────────────────────────────────────────────────────

class RecallAtK(MetricBase):
    """Recall@K = |relevant ∩ retrieved[:K]| / |relevant|"""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    @property
    def name(self) -> str:         return f"Recall@{self.k}"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        if not relevant:
            return 1.0  # nothing to recall
        top = retrieved[: self.k]
        hits = sum(1 for item in top if _is_hit(item, relevant))
        return hits / len(relevant)


# ── F1@K ──────────────────────────────────────────────────────────────────────

class F1AtK(MetricBase):
    """F1@K = harmonic mean of Precision@K and Recall@K"""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._p = PrecisionAtK(k)
        self._r = RecallAtK(k)

    @property
    def name(self) -> str:         return f"F1@{self.k}"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        p = self._p.evaluate(retrieved=retrieved, relevant=relevant)
        r = self._r.evaluate(retrieved=retrieved, relevant=relevant)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


# ── MAP ───────────────────────────────────────────────────────────────────────

class MAPMetric(MetricBase):
    """
    Mean Average Precision.

    AP = (1/R) × Σ P(k)·rel(k)   (Manning 2008, §8.4)
    """

    @property
    def name(self) -> str:         return "MAP"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        if not relevant or not retrieved:
            return 0.0
        num_rel, sum_p = 0, 0.0
        for k, item in enumerate(retrieved, 1):
            if _is_hit(item, relevant):
                num_rel += 1
                sum_p   += num_rel / k
        if num_rel == 0:
            return 0.0
        return sum_p / num_rel


# ── MRR ───────────────────────────────────────────────────────────────────────

class MRRMetric(MetricBase):
    """Mean Reciprocal Rank = 1/rank_of_first_relevant"""

    @property
    def name(self) -> str:         return "MRR"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        for k, item in enumerate(retrieved, 1):
            if _is_hit(item, relevant):
                return 1.0 / k
        return 0.0


# ── NDCG@K ────────────────────────────────────────────────────────────────────

class NDCGAtK(MetricBase):
    """
    Normalized Discounted Cumulative Gain @ K.

    DCG@p  = Σ  rel_i / log₂(i+1)
    NDCG@p = DCG@p / IDCG@p        (Järvelin & Kekäläinen 2002)
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    @property
    def name(self) -> str:         return f"NDCG@{self.k}"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        if not relevant or not retrieved:
            return 0.0

        dcg = sum(
            (1.0 if _is_hit(item, relevant) else 0.0) / math.log2(i + 1)
            for i, item in enumerate(retrieved[: self.k], 1)
        )
        # Ideal DCG: place all relevant items first
        n_ideal = min(len(relevant), self.k)
        idcg    = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

        return dcg / idcg if idcg > 0 else 0.0
