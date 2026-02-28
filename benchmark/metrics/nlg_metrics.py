"""
Natural Language Generation metrics.

References
----------
Papineni, K., et al. (2002). BLEU. ACL, pp. 311–318.
Lin, C. Y. (2004). ROUGE. Text Summarization Branches Out, pp. 74–81.
Banerjee, S., & Lavie, A. (2005). METEOR. ACL workshop, pp. 65–72.

Libraries used: sacrebleu, rouge-score, nltk  (all optional — graceful fallback).
"""

from __future__ import annotations

from typing import Dict

from .base import MetricBase, MetricGroup


# ── Shared lazy-loader ────────────────────────────────────────────────────────

_SACREBLEU  = None
_ROUGE      = None
_NLTK_READY = False


def _load_sacrebleu():
    global _SACREBLEU
    if _SACREBLEU is None:
        try:
            import sacrebleu as _sb
            _SACREBLEU = _sb
        except ImportError:
            _SACREBLEU = False
    return _SACREBLEU or None


def _load_rouge():
    global _ROUGE
    if _ROUGE is None:
        try:
            from rouge_score import rouge_scorer as _rs
            _ROUGE = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        except ImportError:
            _ROUGE = False
    return _ROUGE or None


def _load_nltk():
    global _NLTK_READY
    if not _NLTK_READY:
        try:
            import nltk
            for pkg in ("wordnet", "punkt", "punkt_tab"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            _NLTK_READY = True
        except ImportError:
            pass
    return _NLTK_READY


# ── BLEU ──────────────────────────────────────────────────────────────────────

class BLEUMetric(MetricBase):
    """
    Sentence BLEU score via sacrebleu (returns value in [0, 1]).

    Papineni et al. (2002).
    """

    @property
    def name(self) -> str:         return "BLEU"
    @property
    def group(self) -> MetricGroup: return MetricGroup.NLG
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, reference: str, **_) -> float:
        """
        Args:
            hypothesis: Model-generated answer.
            reference:  Ground-truth answer.
        """
        sb = _load_sacrebleu()
        if sb is None or not hypothesis or not reference:
            return 0.0
        try:
            score = sb.sentence_bleu(hypothesis, [reference])
            return score.score / 100.0      # sacrebleu returns 0–100
        except Exception:
            return 0.0


# ── ROUGE ─────────────────────────────────────────────────────────────────────

class _ROUGEBase(MetricBase):
    """Shared logic for ROUGE-1, ROUGE-2, ROUGE-L."""

    _rouge_key: str = "rouge1"

    @property
    def group(self) -> MetricGroup: return MetricGroup.NLG
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, reference: str, **_) -> float:
        scorer = _load_rouge()
        if scorer is None or not hypothesis or not reference:
            return 0.0
        try:
            scores = scorer.score(reference, hypothesis)
            return scores[self._rouge_key].fmeasure
        except Exception:
            return 0.0


class ROUGE1Metric(_ROUGEBase):
    _rouge_key = "rouge1"
    @property
    def name(self) -> str: return "ROUGE-1"


class ROUGE2Metric(_ROUGEBase):
    _rouge_key = "rouge2"
    @property
    def name(self) -> str: return "ROUGE-2"


class ROUGELMetric(_ROUGEBase):
    _rouge_key = "rougeL"
    @property
    def name(self) -> str: return "ROUGE-L"


# ── METEOR ────────────────────────────────────────────────────────────────────

class METEORMetric(MetricBase):
    """
    METEOR score via nltk.

    Banerjee & Lavie (2005). Requires ``nltk`` + ``wordnet`` corpus.
    Falls back to 0.0 if not available.
    """

    @property
    def name(self) -> str:         return "METEOR"
    @property
    def group(self) -> MetricGroup: return MetricGroup.NLG
    @property
    def requires_llm(self) -> bool: return False
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, hypothesis: str, reference: str, **_) -> float:
        if not _load_nltk() or not hypothesis or not reference:
            return 0.0
        try:
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            return float(meteor_score([word_tokenize(reference)],
                                      word_tokenize(hypothesis)))
        except Exception:
            return 0.0
