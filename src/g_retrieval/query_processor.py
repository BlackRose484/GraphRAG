"""Query expansion, decomposition, and combined expand+extract."""
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
    entities:   dict[str, list[str]]
    query_type: str


class DecomposeAndExtractResult(TypedDict):
    decomposed: list[str]
    entities:   dict[str, list[str]]


class QueryProcessor(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        _logger.info("QueryProcessor initialised")

    def process(self, query: str, *, enable_enhancement: bool = True) -> ProcessedQuery:
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
        except Exception as exc:
            _logger.warning("Query expansion failed: %s", exc)

        try:
            result["decomposed"] = self._decompose(query)
            _logger.info("Query decomposed into %d sub-queries", len(result["decomposed"]))
        except Exception as exc:
            _logger.warning("Query decomposition failed: %s", exc)

        return result

    def expand_and_extract(
        self, query: str, entity_catalog: str = ""
    ) -> ExpandAndExtractResult:
        """Expand query + extract entities + classify query_type in one LLM call."""
        prompt = QUERY_EXPAND_AND_EXTRACT.format(
            query=query,
            entity_catalog=entity_catalog,
        )
        raw = self.safe_generate(prompt).strip()

        if "```" in raw:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                raw = m.group(1)

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

        except (json.JSONDecodeError, Exception) as exc:
            _logger.warning(
                "expand_and_extract: JSON parse failed (%s) — falling back to raw expand",
                exc,
            )
            expanded_fallback = raw.split("\n")[0].strip() or query
            return ExpandAndExtractResult(
                expanded=expanded_fallback,
                entities=dict(_EMPTY_ENTITIES),
                query_type=QueryType.DEFAULT,
            )

    def _expand(self, query: str) -> str:
        prompt = QUERY_EXPAND.format(query=query)
        expanded = self.safe_generate(prompt).strip()
        return expanded or query

    def _decompose(self, query: str) -> list[str]:
        prompt = QUERY_DECOMPOSE.format(query=query)
        response = self.safe_generate(prompt).strip()
        subqueries = [q.strip() for q in response.splitlines() if q.strip()]
        return subqueries or [query]

    def decompose_and_extract(
        self, query: str, entity_catalog: str = ""
    ) -> DecomposeAndExtractResult:
        """Decompose query into sub-queries and extract entities per sub-query in one LLM call."""
        prompt = QUERY_DECOMPOSE_AND_EXTRACT.format(
            query=query,
            entity_catalog=entity_catalog,
        )
        raw = self.safe_generate(prompt).strip()

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

        except (json.JSONDecodeError, ValueError, Exception) as exc:
            _logger.warning(
                "decompose_and_extract: JSON parse failed (%s) — falling back to plain decompose",
                exc,
            )
            return DecomposeAndExtractResult(
                decomposed=self._decompose(query),
                entities=dict(_EMPTY_ENTITIES),
            )
