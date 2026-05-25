"""Weighted score aggregation for the CheoBench benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# Retrieval (1.00) — 60% deterministic IR + 40% LLM-judged context
# Generation (1.00) — 100% LLM-judged answer quality (RAGAs canonical 50/50)
RETRIEVAL_WEIGHTS: Dict[str, float] = {
    # Manning 2008 / Järvelin & Kekäläinen 2002
    "MAP":                    0.15,
    "NDCG@10":                0.15,
    "Precision":              0.10,
    "Recall":                 0.20,
    # Es et al. 2023
    "ContextPrecision":       0.10,
    "ContextRecall":          0.15,
    "ContextEntitiesRecall":  0.15,
}

GENERATION_WEIGHTS: Dict[str, float] = {
    "Faithfulness":          0.50,
    "AnswerRelevance":       0.50,
}

OVERALL_RETRIEVAL_WEIGHT  = 0.5
OVERALL_GENERATION_WEIGHT = 0.5


@dataclass
class CompositeScores:
    s_retrieval:  Optional[float]
    s_generation: Optional[float]
    s_overall:    Optional[float]

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "S_retrieval":  self.s_retrieval,
            "S_generation": self.s_generation,
            "S_overall":    self.s_overall,
        }


def _weighted_average(
    scores:  Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> Optional[float]:
    """Sum weights only over metrics that have a numeric value, then renormalize.

    A missing or ``None`` metric is dropped (not treated as 0) so a disabled
    or errored metric doesn't artificially deflate the composite.
    """
    total_w = 0.0
    total_v = 0.0
    for name, w in weights.items():
        v = scores.get(name)
        if v is None:
            continue
        total_w += w
        total_v += w * v
    if total_w == 0.0:
        return None
    return total_v / total_w


def aggregate(scores: Dict[str, Optional[float]]) -> CompositeScores:
    s_ret = _weighted_average(scores, RETRIEVAL_WEIGHTS)
    s_gen = _weighted_average(scores, GENERATION_WEIGHTS)

    if s_ret is None and s_gen is None:
        s_overall = None
    elif s_ret is None:
        s_overall = s_gen
    elif s_gen is None:
        s_overall = s_ret
    else:
        s_overall = (OVERALL_RETRIEVAL_WEIGHT  * s_ret
                   + OVERALL_GENERATION_WEIGHT * s_gen)

    return CompositeScores(s_retrieval=s_ret, s_generation=s_gen, s_overall=s_overall)
