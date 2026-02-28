"""
Query processing: expansion and decomposition using promt_engineer templates.
"""
from __future__ import annotations

from typing import TypedDict

from src.constants.promt_engineer import QUERY_DECOMPOSE, QUERY_EXPAND
from src.core.base import BaseModel
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ProcessedQuery(TypedDict):
    original: str
    expanded: str
    decomposed: list[str]


class QueryProcessor(BaseModel):
    """Expand and decompose queries for better graph retrieval."""

    def __init__(self) -> None:
        super().__init__()
        _logger.info("QueryProcessor initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, query: str, *, enable_enhancement: bool = True) -> ProcessedQuery:
        """Run the full processing pipeline.

        Args:
            query: Raw user query.
            enable_enhancement: When *False* returns the query unchanged.

        Returns:
            :class:`ProcessedQuery` with original / expanded / decomposed keys.
        """
        result: ProcessedQuery = {
            "original": query,
            "expanded": query,
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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _expand(self, query: str) -> str:
        prompt = QUERY_EXPAND.format(query=query)
        expanded = self.safe_generate(prompt).strip()
        return expanded or query

    def _decompose(self, query: str) -> list[str]:
        prompt = QUERY_DECOMPOSE.format(query=query)
        response = self.safe_generate(prompt).strip()
        subqueries = [q.strip() for q in response.splitlines() if q.strip()]
        return subqueries or [query]
