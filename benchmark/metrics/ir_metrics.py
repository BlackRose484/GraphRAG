"""Information Retrieval metrics.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval.
Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from typing import List, Set

from .base import MetricBase, MetricGroup


# Vietnamese descriptor prefixes that don't change the underlying entity.
# Stripped during normalization so "Vai diễn Thị Mầu" ≡ "Thị Mầu".
_DESCRIPTOR_TOKENS = {
    "vai", "vai diễn", "vai diên", "diễn viên", "dien vien",
    "nhân vật", "nhan vat", "nhan vật", "nhân vat",
    "vở", "vo", "vở chèo", "vo cheo", "trích đoạn", "trich doan",
    "ông", "bà", "anh", "chị", "ong", "ba", "chi",
    "của", "cua", "trong", "cho", "với", "voi",
}

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]+")
_WS_RE    = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _norm(s: str) -> str:
    """Normalize for strict matching; diacritics preserved (use ``_canon`` for fuzzy)."""
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _canon(s: str) -> Set[str]:
    """Canonical word-set for fuzzy entity comparison.

    Returned as a set so word order doesn't matter ("Sùng Bà" ≡ "Bà Sùng")
    and descriptor padding is ignored ("Vai Thị Mầu" ≡ "Thị Mầu").
    """
    s = _strip_diacritics(s.strip().lower())
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in s.split() if t]
    drop = {_strip_diacritics(d) for d in _DESCRIPTOR_TOKENS}
    drop |= {t for d in drop for t in d.split()}
    return {t for t in tokens if t not in drop}


def _same_entity(a_words: Set[str], b_words: Set[str]) -> bool:
    """Subset or Jaccard ≥ 0.6 — fuzzy match for partial overlaps."""
    if not a_words or not b_words:
        return False
    if a_words <= b_words or b_words <= a_words:
        return True
    inter = len(a_words & b_words)
    union = len(a_words | b_words)
    return union > 0 and inter / union >= 0.6


def _is_hit(item: str, relevant: Set[str]) -> bool:
    item_n = _norm(item)
    return any(item_n == _norm(r) or _norm(r) in item_n or item_n in _norm(r)
               for r in relevant)


def _match_relevant(item: str, relevant: Set[str]) -> str | None:
    item_words = _canon(item)
    if not item_words:
        return None
    for r in relevant:
        if _same_entity(item_words, _canon(r)):
            return _norm(r)
    return None


def _hit_flags(retrieved: List[str], relevant: Set[str]) -> List[bool]:
    """Per-position hit flags with no double-count.

    A retrieved item counts as a hit only if it matches a relevant entity
    that has not yet been matched by an earlier retrieved item.
    """
    matched: Set[str] = set()
    flags: List[bool] = []
    for item in retrieved:
        m = _match_relevant(item, relevant)
        if m is not None and m not in matched:
            matched.add(m)
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _dedupe(retrieved: List[str]) -> List[str]:
    """Collapse retrieved items referring to the same entity; first occurrence wins."""
    kept: List[str] = []
    kept_words: List[Set[str]] = []
    for item in retrieved:
        words = _canon(item)
        if not words:
            continue
        if any(_same_entity(words, k) for k in kept_words):
            continue
        kept.append(item)
        kept_words.append(words)
    return kept


class PrecisionMetric(MetricBase):
    """Precision = |relevant ∩ unique(retrieved)| / |unique(retrieved)|.

    No-cutoff form: the downstream LLM consumes all retrieved context as a
    set, so rank-cutoff K is not what determines what the model sees.
    """

    @property
    def name(self) -> str:         return "Precision"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        if not relevant:
            return 1.0
        retrieved = _dedupe(retrieved)
        if not retrieved:
            return 0.0
        flags = _hit_flags(retrieved, relevant)
        return sum(flags) / len(retrieved)


class RecallMetric(MetricBase):
    """Recall = |relevant ∩ unique(retrieved)| / |relevant| (no rank cutoff)."""

    @property
    def name(self) -> str:         return "Recall"
    @property
    def group(self) -> MetricGroup: return MetricGroup.IR
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, retrieved: List[str], relevant: Set[str], **_) -> float:
        if not relevant:
            return 1.0
        retrieved = _dedupe(retrieved)
        flags = _hit_flags(retrieved, relevant)
        return sum(flags) / len(relevant)


class MAPMetric(MetricBase):
    """Mean Average Precision — AP = (1/R) × Σ P(k)·rel(k)  (Manning 2008, §8.4)."""

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
        retrieved = _dedupe(retrieved)
        flags = _hit_flags(retrieved, relevant)
        num_rel, sum_p = 0, 0.0
        for k, hit in enumerate(flags, 1):
            if hit:
                num_rel += 1
                sum_p   += num_rel / k
        if num_rel == 0:
            return 0.0
        return sum_p / len(relevant)


class NDCGAtK(MetricBase):
    """Normalized Discounted Cumulative Gain @ K.

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

        retrieved = _dedupe(retrieved)
        flags = _hit_flags(retrieved[: self.k], relevant)
        dcg = sum(
            (1.0 if hit else 0.0) / math.log2(i + 1)
            for i, hit in enumerate(flags, 1)
        )
        # IDCG: place all relevant items first
        n_ideal = min(len(relevant), self.k)
        idcg    = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

        return dcg / idcg if idcg > 0 else 0.0
