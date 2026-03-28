"""
G-Retrieval orchestrator.

Sequences:
    [enhancement=True]
        ThreadPoolExecutor: expand_and_extract(query) ║ _decompose(query)
            ↓ (both running in parallel)
        LLM-grounded entities (no regex) → GraphRetriever → GraphFormatConverter

    [enhancement=False]
        QueryProcessor.process(pass-through) → EntityExtractor.extract()
            ↓
        GraphRetriever → GraphFormatConverter

Enhancement pipeline (enable_enhancement=True):
    - expand_and_extract: 1 LLM call → expanded query + grounded entities
      (LLM uses the known entity list from the prompt — no regex fallback needed)
    - _decompose: 1 LLM call → focused sub-questions (runs PARALLEL)
    - Entity lists deduplicated before graph retrieval

Optimisation vs legacy:
    Legacy : expand(LLM#1) → decompose(LLM#2) → entity_extract(LLM#3) = 3 sequential calls
    New    : expand+extract(LLM#A) ║ decompose(LLM#B) = 2 parallel calls → ~max(A,B) latency
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any

from src.constants.constant import EntityType, FormatKey, RetrievalMethod
from src.g_retrieval.entity_catalog import EntityCatalog
from src.g_retrieval.community_index import CommunityIndex
from src.g_retrieval.entity_extractor import EntityExtractor
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

    # Extracted entities (for UI display)
    entities: dict[str, list[str]]

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
            "query":            self.query,
            "retrieval_methods": self.retrieval_methods,
            "format_keys":      self.format_keys,
            "num_nodes":        self.num_nodes,
            "num_triplets":     self.num_triplets,
            "num_paths":        self.num_paths,
            "retrieval_time_s": round(self.retrieval_time, 3),
            "error":            self.error,
        }


class RetrievalOrchestrator:
    """Coordinate the full G-Retrieval pipeline.

    Args:
        client: A connected :class:`~src.graph_loader.neo4j_client.Neo4jClient`.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self._entity_catalog   = EntityCatalog()
        self._entity_catalog.load(client)              # 4 lightweight Cypher reads at startup
        self._community_index  = CommunityIndex()
        self._community_index.load(client)             # play-centric subgraph cache
        self._query_processor  = QueryProcessor()
        self._entity_extractor = EntityExtractor()
        self._graph_retriever  = GraphRetriever(client)
        _logger.info("RetrievalOrchestrator initialised (community: %s)",
                     self._community_index.is_loaded())

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

        When *enable_enhancement* is ``True`` two LLM calls run **in parallel**:

        * ``expand_and_extract(query)`` — enriched query + named entities
        * ``_decompose(query)``         — focused sub-questions

        A cheap regex pass over the sub-questions supplements the LLM entities
        at zero extra LLM cost. All entity sets are merged before graph retrieval.

        When *enable_enhancement* is ``False`` the query is used as-is with a
        single EntityExtractor LLM call (no expansion/decomposition overhead).

        Args:
            query:             Raw user query.
            retrieval_methods: Which of nodes/triplets/paths/subgraph to use.
            format_keys:       Which :class:`~src.constants.constant.FormatKey` values to produce.
            enable_enhancement: Toggle parallel expand+extract+decompose pipeline.

        Returns:
            A populated :class:`RetrievalResult`.  On error *error* field is set
            and graph containers are empty.
        """
        retrieval_methods = retrieval_methods or _DEFAULT_METHODS
        format_keys       = format_keys       or _DEFAULT_FORMATS
        start = time.perf_counter()

        try:
            if enable_enhancement:
                processed, entities = self._enhanced_pipeline(query)
            else:
                processed, entities = self._basic_pipeline(query)

            total_entities = sum(len(v) for v in entities.values())
            _logger.info("Retrieval: %d entities total after merge", total_entities)

            # ── Global query fallback ─────────────────────────────────────────
            # If no entities were extracted but the community index is loaded,
            # inject ALL play titles so the community retriever can provide
            # full-KG context for aggregate/global queries.
            if total_entities == 0 and self._community_index.is_loaded():
                all_plays = self._community_index.all_plays()
                if all_plays:
                    entities.setdefault("plays", []).extend(all_plays)
                    _logger.info(
                        "Global fallback: injected %d plays → %s",
                        len(all_plays), all_plays,
                    )

            # ── Graph retrieval ────────────────────────────────────────────────
            _logger.info("Retrieval: querying Neo4j with methods=%s", retrieval_methods)
            graph_data = self._graph_retriever.retrieve(
                entities,
                methods=retrieval_methods,
                community_index=self._community_index,
            )

            # ── Format contexts ────────────────────────────────────────────────
            _logger.info("Retrieval: formatting with keys=%s", format_keys)
            formatted_contexts = GraphFormatConverter.convert_selected(graph_data, format_keys)
            key_facts          = GraphFormatConverter.extract_key_facts(graph_data)

            elapsed = time.perf_counter() - start
            result = RetrievalResult(
                query=query,
                processed_query=processed,
                graph_data=graph_data,
                retrieval_methods=retrieval_methods,
                format_keys=format_keys,
                formatted_contexts=formatted_contexts,
                key_facts=key_facts,
                entities=entities,
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
                "nodes":    [],
                "triplets": [],
                "paths":    [],
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
                entities={},
                num_nodes=0,
                num_triplets=0,
                num_paths=0,
                retrieval_time=elapsed,
                error=str(exc),
            )

    # ── Private pipeline paths ─────────────────────────────────────────────────

    def _enhanced_pipeline(
        self, query: str
    ) -> tuple[ProcessedQuery, dict[str, list[str]]]:
        """Parallel: expand+extract ║ decompose  (2 LLM calls, ~max latency).

        Entity extraction is fully LLM-grounded: the prompt provides the complete
        list of known Cheo entities so no regex fallback is needed.

        Returns (ProcessedQuery, entities_dict).
        """
        _logger.info("Retrieval: launching parallel expand+extract ║ decompose")

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_combined: Future = pool.submit(
                self._query_processor.expand_and_extract,
                query,
                self._entity_catalog.as_text(),    # inject dynamic catalog
            )
            future_decompose: Future = pool.submit(
                self._query_processor._decompose, query
            )
            # Both futures run concurrently; .result() blocks until done
            combined   = future_combined.result()
            decomposed = future_decompose.result()

        entities: dict[str, list[str]] = dict(combined["entities"])

        _logger.info(
            "Retrieval: expand+extract done — %d entities (characters=%d, actors=%d, plays=%d, scenes=%d)",
            sum(len(v) for v in entities.values()),
            len(entities.get("characters", [])),
            len(entities.get("actors", [])),
            len(entities.get("plays", [])),
            len(entities.get("scenes", [])),
        )
        _logger.info("Retrieval: decomposed into %d sub-queries", len(decomposed))

        processed: ProcessedQuery = {
            "original":   query,
            "expanded":   combined["expanded"],
            "decomposed": decomposed,
        }

        return processed, entities

    def _basic_pipeline(
        self, query: str
    ) -> tuple[ProcessedQuery, dict[str, list[str]]]:
        """No enhancement: pass-through query + 1 entity-extract LLM call."""
        _logger.info("Retrieval: enhancement disabled — processing query as-is")
        processed = self._query_processor.process(query, enable_enhancement=False)
        entities  = self._entity_extractor.extract(processed["expanded"])
        return processed, entities
