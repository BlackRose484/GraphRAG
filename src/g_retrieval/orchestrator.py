"""
G-Retrieval orchestrator.

Sequences:
    QueryProcessor → EntityExtractor → GraphRetriever → GraphFormatConverter
and returns a :class:`RetrievalResult` dataclass.

Enhancement pipeline (enable_enhancement=True):
    - QueryProcessor expands the raw query into a richer description.
    - QueryProcessor decomposes it into focused sub-questions.
    - EntityExtractor runs on the *expanded* query (richer context → better recall).
    - Regex scan also runs on every decomposed sub-query at zero extra LLM cost,
      catching entity names that appear only in specific sub-questions.
    - All entity lists are merged & deduplicated before graph retrieval.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.constants.constant import EntityType, FormatKey, RetrievalMethod
from src.g_retrieval.entity_extractor import EntityExtractor, _VN_PROPER_NOUN
from src.g_retrieval.graph_retriever import GraphData, GraphRetriever
from src.g_retrieval.query_processor import ProcessedQuery, QueryProcessor
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.format_converter import GraphFormatConverter
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_DEFAULT_METHODS: list[str] = RetrievalMethod.DEFAULT
_DEFAULT_FORMATS: list[str] = FormatKey.DEFAULT_MID


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_entities(
    base: dict[str, list[str]],
    extra: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return a new entity dict that is the union of *base* and *extra*.

    Values within each category are deduplicated (case-insensitive).
    """
    merged: dict[str, list[str]] = {}
    for key in EntityType.ALL:
        seen_lower: set[str] = set()
        combined: list[str] = []
        for name in [*base.get(key, []), *extra.get(key, [])]:
            low = name.strip().lower()
            if low and low not in seen_lower:
                seen_lower.add(low)
                combined.append(name.strip())
        merged[key] = combined
    return merged


@dataclass
class RetrievalResult:
    """Full output from one retrieval pipeline run."""

    # Input
    query: str
    processed_query: ProcessedQuery

    # Graph data
    graph_data: GraphData

    # Config used
    retrieval_methods: list[str]
    format_keys: list[str]

    # Formatted contexts
    formatted_contexts: dict[str, str]
    key_facts: str

    # Statistics
    num_nodes: int
    num_triplets: int
    num_paths: int
    retrieval_time: float

    # Error (None = success)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        """Compact summary dict suitable for logging / display."""
        return {
            "query": self.query,
            "retrieval_methods": self.retrieval_methods,
            "format_keys": self.format_keys,
            "num_nodes": self.num_nodes,
            "num_triplets": self.num_triplets,
            "num_paths": self.num_paths,
            "retrieval_time_s": round(self.retrieval_time, 3),
            "error": self.error,
        }


class RetrievalOrchestrator:
    """Coordinate the full G-Retrieval pipeline.

    Args:
        client: A connected :class:`~src.graph_loader.neo4j_client.Neo4jClient`.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self._query_processor = QueryProcessor()
        self._entity_extractor = EntityExtractor()
        self._graph_retriever = GraphRetriever(client)
        _logger.info("RetrievalOrchestrator initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        retrieval_methods: list[str] | None = None,
        format_keys: list[str] | None = None,
        enable_enhancement: bool = True,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline.

        When *enable_enhancement* is ``True`` the entity extraction receives the
        **expanded** query (richer context → higher entity recall), and a cheap
        regex pass runs over every decomposed sub-query to catch entity names
        that appear only in specific sub-questions — all at zero extra LLM cost.
        All entity sets are merged before graph retrieval.

        Args:
            query: Raw user query.
            retrieval_methods: Which of nodes/triplets/paths/subgraph to use.
            format_keys: Which :class:`~src.constants.constant.FormatKey` values to produce.
            enable_enhancement: Forward to :class:`~.query_processor.QueryProcessor`.

        Returns:
            A populated :class:`RetrievalResult`.  On error *error* field is set
            and graph containers are empty.
        """
        retrieval_methods = retrieval_methods or _DEFAULT_METHODS
        format_keys = format_keys or _DEFAULT_FORMATS
        start = time.perf_counter()

        try:
            # ── 1. Query processing ────────────────────────────────────────────
            _logger.info("Retrieval: processing query")
            processed = self._query_processor.process(query, enable_enhancement=enable_enhancement)

            # ── 2. Entity extraction ───────────────────────────────────────────
            # Feed the *expanded* query to the LLM extractor (richer context).
            # If enhancement is off, processed.expanded == query (no change).
            _logger.info("Retrieval: extracting entities from expanded query")
            entities = self._entity_extractor.extract(processed["expanded"])

            # When enhancement is ON, also regex-scan every sub-query so entities
            # mentioned only in specific decomposed questions are not missed.
            # This costs zero extra LLM calls.
            if enable_enhancement:
                sub_queries = processed.get("decomposed", [])
                # skip sub-queries identical to the expanded query (already covered)
                novel_subs = [
                    sq for sq in sub_queries
                    if sq.strip() and sq.strip() != processed["expanded"].strip()
                ]
                if novel_subs:
                    regex_names: list[str] = []
                    for sq in novel_subs:
                        regex_names.extend(_VN_PROPER_NOUN.findall(sq))
                    if regex_names:
                        entities = _merge_entities(
                            entities,
                            {
                                EntityType.CHARACTERS: list(set(regex_names)),
                                EntityType.ACTORS:     [],
                                EntityType.PLAYS:      [],
                                EntityType.SCENES:     [],
                            },
                        )
                        _logger.info(
                            "Retrieval: merged %d regex entities from %d sub-queries",
                            len(regex_names), len(novel_subs),
                        )

            total_entities = sum(len(v) for v in entities.values())
            _logger.info("Retrieval: %d entities total after merge", total_entities)

            # ── 3. Graph retrieval ─────────────────────────────────────────────
            _logger.info("Retrieval: querying Neo4j with methods=%s", retrieval_methods)
            graph_data = self._graph_retriever.retrieve(entities, methods=retrieval_methods)

            # ── 4. Format contexts ─────────────────────────────────────────────
            _logger.info("Retrieval: formatting with keys=%s", format_keys)
            formatted_contexts = GraphFormatConverter.convert_selected(graph_data, format_keys)
            key_facts = GraphFormatConverter.extract_key_facts(graph_data)

            elapsed = time.perf_counter() - start
            result = RetrievalResult(
                query=query,
                processed_query=processed,
                graph_data=graph_data,
                retrieval_methods=retrieval_methods,
                format_keys=format_keys,
                formatted_contexts=formatted_contexts,
                key_facts=key_facts,
                num_nodes=len(graph_data.get("nodes", [])),
                num_triplets=len(graph_data.get("triplets", [])),
                num_paths=len(graph_data.get("paths", [])),
                retrieval_time=elapsed,
            )
            _logger.info("Retrieval complete: %s", result.summary())
            return result

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            _logger.error("Retrieval pipeline failed: %s", exc, exc_info=True)
            empty_graph: GraphData = {
                "nodes": [],
                "triplets": [],
                "paths": [],
                "subgraph": {"nodes": [], "relationships": []},
            }
            return RetrievalResult(
                query=query,
                processed_query={"original": query, "expanded": query, "decomposed": [query]},
                graph_data=empty_graph,
                retrieval_methods=retrieval_methods,
                format_keys=format_keys,
                formatted_contexts={},
                key_facts="",
                num_nodes=0,
                num_triplets=0,
                num_paths=0,
                retrieval_time=elapsed,
                error=str(exc),
            )
