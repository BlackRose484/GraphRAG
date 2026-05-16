"""
Query processing: expansion, decomposition, and combined expand+extract.

Public methods
--------------
process(query, enable_enhancement)
    Legacy pipeline (sequential expand → decompose). Used when enhancement=False.

expand_and_extract(query) -> dict
    Combined single-LLM-call: returns {expanded, entities} simultaneously.
    Designed to run in parallel with _decompose() via ThreadPoolExecutor.

_decompose(query) -> list[str]
    Kept public-ish so orchestrator can submit it to a thread pool directly.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from src.constants.constant import EntityType, QueryType
from src.constants.prompt_engineer import (
    QUERY_DECOMPOSE,
    QUERY_DECOMPOSE_AND_EXTRACT,
    QUERY_EXPAND,
    QUERY_EXPAND_AND_EXTRACT,
)
from src.core.base import BaseModel
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_EMPTY_ENTITIES: dict[str, list[str]] = {
    EntityType.CHARACTERS: [],
    EntityType.ACTORS:     [],
    EntityType.PLAYS:      [],
    EntityType.SCENES:     [],
}


class ProcessedQuery(TypedDict):
    original:   str
    expanded:   str
    decomposed: list[str]


class ExpandAndExtractResult(TypedDict):
    expanded:   str
    entities:   dict[str, list[str]]   # keys: characters, actors, plays, scenes
    query_type: str                    # one of QueryType.ALL ("Local"/"Community"/"Global")


class DecomposeAndExtractResult(TypedDict):
    """Output of Tác vụ B — phân rã + trích xuất bổ sung trên các sub-query."""
    decomposed: list[str]              # the sub-query strings (for UI/debug)
    entities:   dict[str, list[str]]   # union of entities across all sub-queries


class QueryProcessor(BaseModel):
    """Expand and decompose queries for better graph retrieval.

    Two usage modes:

    **Legacy (sequential)**::

        processed = processor.process(query)          # 2 LLM calls

    **Optimised (parallel-friendly)**::

        # Call these two concurrently via ThreadPoolExecutor:
        combined   = processor.expand_and_extract(query)  # LLM call A
        decomposed = processor._decompose(query)           # LLM call B
        # → total latency ≈ max(A, B) instead of A + B
    """

    def __init__(self) -> None:
        super().__init__()
        _logger.info("QueryProcessor initialised")

    # ── Public API — Legacy ────────────────────────────────────────────────────

    def process(self, query: str, *, enable_enhancement: bool = True) -> ProcessedQuery:
        """Run the full sequential processing pipeline.

        Args:
            query: Raw user query.
            enable_enhancement: When *False* returns the query unchanged
                                (skips both LLM calls).

        Returns:
            :class:`ProcessedQuery` with original / expanded / decomposed keys.
        """
        result: ProcessedQuery = {
            "original":   query,
            "expanded":   query,
            "decomposed": [query],
        }

        if not enable_enhancement:
            _logger.debug("Query enhancement disabled — returning original query")
            return result

        try:
            result["expanded"] = self._expand(query)
            _logger.info("Query expansion succeeded")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Query expansion failed: %s", exc)

        try:
            result["decomposed"] = self._decompose(query)
            _logger.info("Query decomposed into %d sub-queries", len(result["decomposed"]))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Query decomposition failed: %s", exc)

        return result

    # ── Public API — Optimised (parallel-friendly) ─────────────────────────────

    def expand_and_extract(
        self, query: str, entity_catalog: str = ""
    ) -> ExpandAndExtractResult:
        """Single LLM call: expand *query* and extract named entities at once.

        Replaces the sequential ``_expand()`` (LLM call #1) +
        ``EntityExtractor._extract_by_llm()`` (LLM call #3) with one combined
        call. Designed to be submitted to a ``ThreadPoolExecutor`` alongside
        ``_decompose()`` so both run in parallel.

        Args:
            query:          Raw (or original) user query.
            entity_catalog: Formatted entity list string from
                            :class:`~src.g_retrieval.entity_catalog.EntityCatalog`.
                            Injected into the ``{entity_catalog}`` placeholder
                            in :data:`~src.constants.prompt_engineer.QUERY_EXPAND_AND_EXTRACT`.
                            When empty the LLM falls back to its own Cheo knowledge.

        Returns:
            Dict with:
              - ``expanded``: enriched query string
              - ``entities``: ``{characters, actors, plays, scenes}`` lists
        """
        prompt = QUERY_EXPAND_AND_EXTRACT.format(
            query=query,
            entity_catalog=entity_catalog,
        )
        raw = self.safe_generate(prompt).strip()


        # ── Parse JSON from LLM response ──────────────────────────────────────
        # Strip ```json ... ``` fences if present
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                raw = m.group(1)

        # Extract bare JSON object
        m2 = re.search(r"\{.*\}", raw, re.DOTALL)
        if m2:
            raw = m2.group(0)

        try:
            parsed: dict[str, Any] = json.loads(raw)

            expanded = (parsed.get("expanded") or "").strip() or query

            raw_ents = parsed.get("entities", {})
            entities: dict[str, list[str]] = {
                key: [str(v) for v in raw_ents.get(key, [])]
                if isinstance(raw_ents.get(key), list)
                else []
                for key in EntityType.ALL
            }

            # Parse query_type with case-insensitive normalisation and safe fallback
            raw_qt = str(parsed.get("query_type", "")).strip()
            query_type = next(
                (qt for qt in QueryType.ALL if qt.lower() == raw_qt.lower()),
                "",
            )
            if not query_type:
                _logger.warning(
                    "expand_and_extract: invalid query_type %r — fallback to %s",
                    raw_qt, QueryType.DEFAULT,
                )
                query_type = QueryType.DEFAULT

            total = sum(len(v) for v in entities.values())
            _logger.info(
                "expand_and_extract: expanded OK, %d entities extracted, query_type=%s",
                total, query_type,
            )
            return ExpandAndExtractResult(
                expanded=expanded,
                entities=entities,
                query_type=query_type,
            )

        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
            _logger.warning(
                "expand_and_extract: JSON parse failed (%s) — falling back to raw expand",
                exc,
            )
            # Graceful fallback: try to at least return a non-empty expanded string
            expanded_fallback = raw.split("\n")[0].strip() or query
            return ExpandAndExtractResult(
                expanded=expanded_fallback,
                entities=dict(_EMPTY_ENTITIES),
                query_type=QueryType.DEFAULT,
            )

    # ── Private / thread-pool helpers ──────────────────────────────────────────

    def _expand(self, query: str) -> str:
        """Expand *query* with Cheo-domain context (1 LLM call)."""
        prompt = QUERY_EXPAND.format(query=query)
        expanded = self.safe_generate(prompt).strip()
        return expanded or query

    def _decompose(self, query: str) -> list[str]:
        """Break *query* into focused sub-questions (1 LLM call).

        Legacy decomposition without entity extraction. Kept for the basic
        pipeline mode; the enhanced pipeline uses
        :meth:`decompose_and_extract` instead so Tác vụ B contributes entities
        rather than just sub-question strings.
        """
        prompt = QUERY_DECOMPOSE.format(query=query)
        response = self.safe_generate(prompt).strip()
        subqueries = [q.strip() for q in response.splitlines() if q.strip()]
        return subqueries or [query]

    def decompose_and_extract(
        self, query: str, entity_catalog: str = ""
    ) -> DecomposeAndExtractResult:
        """Single LLM call: decompose *query* AND extract entities from each sub-query.

        Tác vụ B của pipeline song song. Khác với :meth:`_decompose` legacy,
        method này chạy entity extraction ngay trên từng sub-query — góc nhìn
        bổ sung so với Tác vụ A (chỉ extract trên *query* gốc). Kết quả được
        gộp với entities của Tác vụ A để cải thiện recall cho câu hỏi đa khía
        cạnh.

        Args:
            query:          Raw user query.
            entity_catalog: Formatted entity list, injected vào prompt để
                            extraction được catalog-grounded — cùng cơ chế
                            ràng buộc định danh với Tác vụ A.

        Returns:
            ``{decomposed: [...], entities: {...}}`` — danh sách sub-query
            (cho UI/debug) và tập entities hợp nhất (union, dedup) trên các
            sub-query.
        """
        prompt = QUERY_DECOMPOSE_AND_EXTRACT.format(
            query=query,
            entity_catalog=entity_catalog,
        )
        raw = self.safe_generate(prompt).strip()

        # Strip ```json ... ``` fences if present
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                raw = m.group(1)
        m2 = re.search(r"\{.*\}", raw, re.DOTALL)
        if m2:
            raw = m2.group(0)

        try:
            parsed: dict[str, Any] = json.loads(raw)
            items = parsed.get("decomposed", [])
            if not isinstance(items, list):
                raise ValueError("decomposed must be a list")

            decomposed: list[str] = []
            merged: dict[str, list[str]] = {k: [] for k in EntityType.ALL}
            seen_lower: dict[str, set[str]] = {k: set() for k in EntityType.ALL}

            for item in items:
                if not isinstance(item, dict):
                    continue
                q_text = str(item.get("question", "")).strip()
                if q_text:
                    decomposed.append(q_text)
                ents = item.get("entities", {}) or {}
                for key in EntityType.ALL:
                    raw_list = ents.get(key, [])
                    if not isinstance(raw_list, list):
                        continue
                    for v in raw_list:
                        name = str(v).strip()
                        low = name.lower()
                        if name and low not in seen_lower[key]:
                            seen_lower[key].add(low)
                            merged[key].append(name)

            if not decomposed:
                decomposed = [query]

            total = sum(len(v) for v in merged.values())
            _logger.info(
                "decompose_and_extract: %d sub-queries, %d entities (union)",
                len(decomposed), total,
            )
            return DecomposeAndExtractResult(
                decomposed=decomposed,
                entities=merged,
            )

        except (json.JSONDecodeError, ValueError, Exception) as exc:  # noqa: BLE001
            _logger.warning(
                "decompose_and_extract: JSON parse failed (%s) — falling back to plain decompose",
                exc,
            )
            return DecomposeAndExtractResult(
                decomposed=self._decompose(query),
                entities=dict(_EMPTY_ENTITIES),
            )
