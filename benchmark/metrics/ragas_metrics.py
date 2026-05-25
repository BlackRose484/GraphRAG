"""RAGAS-inspired LLM-based metrics.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated
Evaluation of Retrieval Augmented Generation. arXiv:2309.15217
"""

from __future__ import annotations

import re
from typing import List

from .base import MetricBase, MetricGroup


class _RAGASBase(MetricBase):
    """Shared scaffolding for RAGAS metrics.

    All RAGAs LLM calls use ``temperature=0`` so judge scores are reproducible
    across benchmark runs (variance from LLM stochasticity would invalidate
    cross-pipeline comparisons).
    """

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
    def requires_ground_truth(self) -> bool: return False

    def _ask(self, prompt: str) -> str:
        return self._llm.safe_generate(prompt, temperature=0.0)

    @staticmethod
    def _parse_float(text: str, fallback: float = 0.0) -> float:
        m = re.search(r"\b([01](?:\.\d+)?)\b", text)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        nums = re.findall(r"\d+\.?\d*", text)
        if nums:
            v = float(nums[0])
            return max(0.0, min(1.0, v / 10.0 if v > 1 else v))
        return fallback

    @staticmethod
    def _extract_list(text: str) -> List[str]:
        lines = [re.sub(r"^\s*[\d\-\*\.]+\s*", "", l).strip()
                 for l in text.splitlines()]
        return [l for l in lines if len(l) > 5]


class FaithfulnessMetric(_RAGASBase):
    """Faithfulness = supported claims / total claims in answer.

    Two-step: (1) LLM extracts atomic claims, (2) LLM verifies each against context.
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

        claims_raw = self._ask(
            f"Liệt kê từng mệnh đề nguyên tử (atomic claim) trong câu trả lời sau. "
            f"Mỗi mệnh đề trên một dòng.\n\nCâu trả lời:\n{hypothesis}"
        )
        claims = self._extract_list(claims_raw)
        if not claims:
            return 1.0

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


class AnswerRelevanceMetric(_RAGASBase):
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


class ContextPrecisionMetric(_RAGASBase):
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


class ContextRecallMetric(_RAGASBase):
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


class ContextRelevanceMetric(_RAGASBase):
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


class ContextEntitiesRecallMetric(_RAGASBase):
    """Fraction of ground-truth entities appearing in the retrieved context.

    Distinct from ``EntityCoverage`` (which checks the generated answer) —
    this measures whether the retriever surfaced the entities the gold answer
    relies on, regardless of whether the LLM later used them.
    """

    @property
    def name(self) -> str: return "ContextEntitiesRecall"
    @property
    def requires_ground_truth(self) -> bool: return True

    def evaluate(self, reference: str, context: str, entities: List[str] = None, **_) -> float:
        if not reference or not context:
            return 0.0

        if entities:
            ent_list = list(entities)
        else:
            extracted = self._ask(
                f"Liệt kê các thực thể (tên người, vở chèo, vai diễn, sự kiện) "
                f"có trong câu trả lời sau. Mỗi thực thể trên một dòng, "
                f"chỉ ghi tên thực thể, không thêm chú thích.\n\n"
                f"Câu trả lời: {reference}"
            )
            ent_list = self._extract_list(extracted)
            if not ent_list:
                return 0.0

        ctx_lower = context.lower()
        found = sum(1 for e in ent_list if e and e.strip().lower() in ctx_lower)
        return found / len(ent_list)
