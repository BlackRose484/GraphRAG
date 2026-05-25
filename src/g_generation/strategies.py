"""G-Generation strategies: Pre, Mid, and Post."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.constants.constant import FormatKey, GenerationStrategy
from src.constants.prompt_engineer import (
    CONTEXT_HEADER,
    CONTEXT_KEY_FACTS_HEADER,
    CONTEXT_SECTION,
    MID_GENERATION,
    POST_INITIAL,
    POST_REFINE,
    PRE_GENERATION,
)
from src.core.base import BaseModel
from src.utils.format_converter import GraphFormatConverter
from src.utils.logger import get_logger

_logger = get_logger(__name__)

GraphData = dict[str, Any]


class BaseGenerationStrategy(ABC):
    """Abstract base for Pre / Mid / Post strategies."""

    def __init__(self) -> None:
        self._model = BaseModel()

    @abstractmethod
    def generate(
        self,
        query: str,
        graph_data: GraphData,
        selected_formats: list[str] | None = None,
        key_facts: str = "",
    ) -> str:
        """Return the generated answer string."""

    def _build_context(
        self,
        graph_data: GraphData,
        selected_formats: list[str],
        key_facts: str = "",
    ) -> str:
        parts: list[str] = [CONTEXT_HEADER]

        kf = key_facts or GraphFormatConverter.extract_key_facts(graph_data)
        parts.append(CONTEXT_KEY_FACTS_HEADER.format(key_facts=kf))

        converters = {
            FormatKey.NATURAL_LANGUAGE: GraphFormatConverter.to_natural_language,
            FormatKey.ADJACENCY_TABLE:  GraphFormatConverter.to_adjacency_table,
            FormatKey.CODE_LIKE:        GraphFormatConverter.to_code_like,
            FormatKey.NODE_SEQUENCE:    GraphFormatConverter.to_node_sequence,
        }

        for fmt in selected_formats:
            conv = converters.get(fmt)
            if conv is None:
                _logger.warning("Unknown format key '%s' — skipped", fmt)
                continue
            try:
                content = conv(graph_data)
                title = FormatKey.TITLES.get(fmt, fmt)
                parts.append(CONTEXT_SECTION.format(title=title, content=content))
            except Exception as exc:
                _logger.warning("Format conversion failed for '%s': %s", fmt, exc)

        return "\n".join(parts)


class PreGenerationStrategy(BaseGenerationStrategy):
    """Inject full graph context before the question."""

    DEFAULT_FORMATS = FormatKey.DEFAULT_PRE

    def generate(
        self,
        query: str,
        graph_data: GraphData,
        selected_formats: list[str] | None = None,
        key_facts: str = "",
        *,
        prebuilt_context: str = "",
    ) -> str:
        formats = selected_formats or self.DEFAULT_FORMATS
        context = prebuilt_context or self._build_context(graph_data, formats, key_facts=key_facts)
        prompt = PRE_GENERATION.format(context=context, query=query)
        response = self._model.safe_generate(prompt)
        _logger.info("PreGenerationStrategy: generation complete")
        return response


class MidGenerationStrategy(BaseGenerationStrategy):
    """Structured mid-generation guidance."""

    DEFAULT_FORMATS = FormatKey.DEFAULT_MID

    def generate(
        self,
        query: str,
        graph_data: GraphData,
        selected_formats: list[str] | None = None,
        key_facts: str = "",
        *,
        prebuilt_context: str = "",
    ) -> str:
        formats = selected_formats or self.DEFAULT_FORMATS
        kf = key_facts or GraphFormatConverter.extract_key_facts(graph_data)
        context = prebuilt_context or self._build_context(graph_data, formats, key_facts=kf)
        prompt = MID_GENERATION.format(context=context, query=query, key_facts=kf)
        response = self._model.safe_generate(prompt)
        _logger.info("MidGenerationStrategy: generation complete")
        return response


class PostGenerationStrategy(BaseGenerationStrategy):
    """Two-step: rough answer → graph-verified refinement."""

    DEFAULT_FORMATS = FormatKey.DEFAULT_POST

    def generate(
        self,
        query: str,
        graph_data: GraphData,
        selected_formats: list[str] | None = None,
        key_facts: str = "",
        *,
        prebuilt_context: str = "",
    ) -> str:
        formats = selected_formats or self.DEFAULT_FORMATS

        initial_prompt = POST_INITIAL.format(query=query)
        initial_answer = self._model.safe_generate(initial_prompt)
        _logger.info("PostGenerationStrategy: initial answer generated")

        kf = key_facts or GraphFormatConverter.extract_key_facts(graph_data)
        context = prebuilt_context or self._build_context(graph_data, formats, key_facts=kf)
        verification_data = f"=== KEY FACTS ===\n{kf}\n\n{context}"
        refine_prompt = POST_REFINE.format(
            query=query,
            initial_answer=initial_answer,
            verification_data=verification_data,
        )
        refined = self._model.safe_generate(refine_prompt)
        _logger.info("PostGenerationStrategy: refinement complete")
        return refined


STRATEGY_REGISTRY: dict[str, type[BaseGenerationStrategy]] = {
    GenerationStrategy.PRE:  PreGenerationStrategy,
    GenerationStrategy.MID:  MidGenerationStrategy,
    GenerationStrategy.POST: PostGenerationStrategy,
}
