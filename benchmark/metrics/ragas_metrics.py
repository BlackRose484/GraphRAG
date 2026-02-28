"""
RAGAS-inspired LLM-based metrics.

Reference
---------
Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).
    RAGAS: Automated Evaluation of Retrieval Augmented Generation.
    arXiv:2309.15217

All five metrics use :class:`src.core.base.BaseModel` (LiteLLM) so the same
LLM configured in ``.env`` is used — no extra SDK needed.

These metrics are **disabled by default** in :class:`MetricRegistry` because
each question triggers 1–3 LLM calls.
"""

from __future__ import annotations

import re
from typing import List

from .base import MetricBase, MetricGroup


# ── Shared LLM helper ─────────────────────────────────────────────────────────

class _RAGASBase(MetricBase):
    """Shared scaffolding for RAGAS metrics."""

    def __init__(self) -> None:
        from src.core.base import BaseModel

        class _M(BaseModel):
            pass

        self._llm = _M()

    @property
    def group(self) -> MetricGroup: return MetricGroup.RAGAS
    @property
    def requires_llm(self) -> bool: return True
    @property
    def requires_ground_truth(self) -> bool: return False  # overridden where needed

    def _ask(self, prompt: str) -> str:
        return self._llm.safe_generate(prompt)

    @staticmethod
    def _parse_float(text: str, fallback: float = 0.0) -> float:
        m = re.search(r"\b([01](?:\.\d+)?)\b", text)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        # Try extracting any decimal in text
        nums = re.findall(r"\d+\.?\d*", text)
        if nums:
            v = float(nums[0])
            return max(0.0, min(1.0, v / 10.0 if v > 1 else v))
        return fallback

    @staticmethod
    def _extract_list(text: str) -> List[str]:
        """Extract newline or numbered list items from LLM output."""
        lines = [re.sub(r"^\s*[\d\-\*\.]+\s*", "", l).strip()
                 for l in text.splitlines()]
        return [l for l in lines if len(l) > 5]


# ── Faithfulness ──────────────────────────────────────────────────────────────

class FaithfulnessMetric(_RAGASBase):
    """
    Faithfulness = supported claims / total claims in answer.

    Method:
        1. LLM extracts atomic claims from the answer.
        2. LLM verifies each claim against the retrieved context.
        3. Score = fraction of supported claims.
    """

    @property
    def name(self) -> str: return "Faithfulness"

    def evaluate(
        self,
        hypothesis:  str,
        context:     str,
        **_,
    ) -> float:
        if not hypothesis or not context:
            return 0.0

        # Step 1 — extract claims
        claims_raw = self._ask(
            f"Liệt kê từng mệnh đề nguyên tử (atomic claim) trong câu trả lời sau. "
            f"Mỗi mệnh đề trên một dòng.\n\nCâu trả lời:\n{hypothesis}"
        )
        claims = self._extract_list(claims_raw)
        if not claims:
            return 1.0

        # Step 2 — verify each claim
        supported = 0
        for claim in claims:
            verdict = self._ask(
                f"Ngữ cảnh:\n{context}\n\n"
                f"Mệnh đề: {claim}\n\n"
                f"Mệnh đề này có được hỗ trợ bởi ngữ cảnh không? "
                f"Trả lời chỉ với 'yes' hoặc 'no'."
            )
            if "yes" in verdict.lower():
                supported += 1

        return supported / len(claims)


# ── Answer Relevance ──────────────────────────────────────────────────────────

class AnswerRelevanceMetric(_RAGASBase):
    """
    Answer Relevance — how well does the answer address the question?

    Prompt LLM to score directly on [0, 1].
    """

    @property
    def name(self) -> str: return "AnswerRelevance"

    def evaluate(self, question: str, hypothesis: str, **_) -> float:
        if not question or not hypothesis:
            return 0.0
        raw = self._ask(
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời: {hypothesis}\n\n"
            f"Đánh giá mức độ câu trả lời giải quyết câu hỏi trên thang điểm 0.0 đến 1.0. "
            f"Chỉ trả về một số thập phân."
        )
        return self._parse_float(raw)


# ── Context Precision ─────────────────────────────────────────────────────────

class ContextPrecisionMetric(_RAGASBase):
    """
    Context Precision — fraction of retrieved context chunks that are relevant
    to the question.

    Prompt LLM to rate relevance of the context.
    """

    @property
    def name(self) -> str: return "ContextPrecision"

    def evaluate(self, question: str, context: str, **_) -> float:
        if not question or not context:
            return 0.0
        raw = self._ask(
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh được truy xuất:\n{context}\n\n"
            f"Tỉ lệ ngữ cảnh trên thực sự liên quan đến câu hỏi là bao nhiêu? "
            f"Chỉ trả về một số từ 0.0 đến 1.0."
        )
        return self._parse_float(raw)


# ── Context Recall ────────────────────────────────────────────────────────────

class ContextRecallMetric(_RAGASBase):
    """
    Context Recall — how much of the ground-truth answer is covered by the
    retrieved context?
    """

    @property
    def name(self) -> str: return "ContextRecall"
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, reference: str, context: str, **_) -> float:
        if not reference or not context:
            return 0.0
        raw = self._ask(
            f"Câu trả lời chuẩn: {reference}\n\n"
            f"Ngữ cảnh được truy xuất:\n{context}\n\n"
            f"Bao nhiêu phần của câu trả lời chuẩn được đề cập trong ngữ cảnh? "
            f"Trả lời chỉ một số từ 0.0 đến 1.0."
        )
        return self._parse_float(raw)


# ── Context Relevance ─────────────────────────────────────────────────────────

class ContextRelevanceMetric(_RAGASBase):
    """
    Context Relevance — overall relevance of the retrieved context to the
    question, independent of the generated answer.
    """

    @property
    def name(self) -> str: return "ContextRelevance"

    def evaluate(self, question: str, context: str, **_) -> float:
        if not question or not context:
            return 0.0
        raw = self._ask(
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Ngữ cảnh này liên quan đến câu hỏi ở mức độ nào? "
            f"Chỉ trả về một số từ 0.0 đến 1.0."
        )
        return self._parse_float(raw)
