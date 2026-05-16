"""
G-Retrieval orchestrator — two retrieval paths sharing one final stage.

ENHANCED path (``enable_enhancement=True`` — default, used in production)
    ThreadPoolExecutor: expand_and_extract(query) ║ decompose_and_extract(query)
        ↓ 2 LLM calls running in parallel (~max(A,B) latency)
    Union catalog-grounded entities → query_type → auto-route methods/strategy
        ↓
    GraphRetriever → GraphFormatConverter

    Used by:  pages 🔍 GraphRAG, ⚖️ Compare, 💬 Chat, 📊 Benchmark.
    Why     : maximum recall + query-type classification enables auto-routing.

BASIC path (``enable_enhancement=False`` — for user-study fair comparison)
    QueryProcessor.process(pass-through, no LLM)
        ↓
    EntityExtractor.extract()  ← 1 LLM call, NO catalog grounding
        ↓
    query_type = "Community" (hard default, no LLM classification)
        ↓
    GraphRetriever → GraphFormatConverter

    Used by:  page 🧪 Experiment (user study).
    Why     : Vector RAG baseline has no query enhancement; disabling it on
              GraphRAG too keeps the comparison strictly "graph retrieval vs
              vector retrieval", not "graph + LLM rewrites vs vector".
              This is a deliberate scientific choice, NOT deprecated.

Both paths converge at GraphRetriever — the same Cypher methods + format
converters are used regardless of which path produced the entity list.

Latency contrast:
    ENHANCED : expand+extract(LLM-A) ║ decompose+extract(LLM-B) → ~max(A,B)
    BASIC    : entity_extract(LLM-C)                            → ~C
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any

from src.constants.constant import EntityType, FormatKey, QueryType, RetrievalMethod
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

    # Query classification (used by auto-routing)
    query_type: str

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
            "query_type":       self.query_type,
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
        auto_routing: bool = True,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline.

        When *enable_enhancement* is ``True`` two LLM calls run **in parallel**:

        * ``expand_and_extract(query)`` — enriched query + named entities + query_type
        * ``_decompose(query)``         — focused sub-questions

        When *enable_enhancement* is ``False`` the query is used as-is with a
        single EntityExtractor LLM call (no expansion/decomposition overhead);
        in that mode ``query_type`` defaults to :attr:`QueryType.DEFAULT`.

        Auto-routing
        ------------
        When ``auto_routing=True`` AND the caller did not pass an explicit
        ``retrieval_methods`` list, the activated methods are derived from the
        query type via :data:`QueryType.METHODS`:

            Local     → [nodes, triplets]
            Community → [nodes, triplets, paths]
            Global    → [nodes, triplets, paths, subgraph]

        When ``auto_routing=False`` (or an explicit list is supplied) the
        caller's choice is respected — used by benchmark runs that need to
        measure each retrieval configuration in isolation.

        Args:
            query:              Raw user query.
            retrieval_methods:  Explicit method list; overrides auto-routing.
            format_keys:        Which :class:`~src.constants.constant.FormatKey` values to produce.
            enable_enhancement: Toggle parallel expand+extract ║ decompose pipeline.
            auto_routing:       Let the LLM-classified query_type pick methods.

        Returns:
            A populated :class:`RetrievalResult`.  On error *error* field is set
            and graph containers are empty.
        """
        explicit_methods = retrieval_methods is not None
        format_keys      = format_keys or _DEFAULT_FORMATS
        start = time.perf_counter()

        try:
            if enable_enhancement:
                processed, entities, query_type = self._enhanced_pipeline(query)
            else:
                processed, entities, query_type = self._basic_pipeline(query)

            # Auto-route: pick methods from query_type when caller didn't override
            if not explicit_methods and auto_routing:
                retrieval_methods = list(QueryType.METHODS[query_type])
                _logger.info(
                    "Auto-routing: query_type=%s → methods=%s",
                    query_type, retrieval_methods,
                )
            elif retrieval_methods is None:
                retrieval_methods = list(_DEFAULT_METHODS)

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
                query_type=query_type,
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
                retrieval_methods=retrieval_methods or list(_DEFAULT_METHODS),
                format_keys=format_keys,
                formatted_contexts={},
                key_facts="",
                entities={},
                query_type=QueryType.DEFAULT,
                num_nodes=0,
                num_triplets=0,
                num_paths=0,
                retrieval_time=elapsed,
                error=str(exc),
            )

    # ── Private pipeline paths ─────────────────────────────────────────────────

    def _enhanced_pipeline(
        self, query: str
    ) -> tuple[ProcessedQuery, dict[str, list[str]], str]:
        """Parallel multi-query pipeline (2 LLM calls, ~max latency).

        Two LLM tasks run concurrently:

        * **Tác vụ A** — :meth:`expand_and_extract`: trên câu hỏi gốc *q* —
          mở rộng, trích xuất entities và phân loại Local/Community/Global.
        * **Tác vụ B** — :meth:`decompose_and_extract`: phân rã *q* thành
          các sub-query và trích xuất entities trên từng sub-query.

        Both tasks inject the entity catalog so extraction is catalog-grounded.
        After both complete, entities from A and B are unioned (deduplicated)
        — A provides precision (focused on the original phrasing), B provides
        recall (catches entities surfaced by alternative framings of the
        question).

        Returns (ProcessedQuery, merged_entities, query_type).
        """
        _logger.info("Retrieval: launching parallel Tác vụ A ║ Tác vụ B")

        catalog_text = self._entity_catalog.as_text()

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a: Future = pool.submit(
                self._query_processor.expand_and_extract,
                query, catalog_text,
            )
            future_b: Future = pool.submit(
                self._query_processor.decompose_and_extract,
                query, catalog_text,
            )
            # Both futures run concurrently; .result() blocks until done
            result_a = future_a.result()
            result_b = future_b.result()

        entities_a:  dict[str, list[str]] = dict(result_a["entities"])
        entities_b:  dict[str, list[str]] = dict(result_b["entities"])
        decomposed:  list[str]            = list(result_b["decomposed"])
        query_type:  str                  = result_a["query_type"]

        # Union entities from both tasks — A gives precision, B gives recall
        merged_entities = _merge_entities(entities_a, entities_b)

        _logger.info(
            "Retrieval: Tác vụ A → %d entities, query_type=%s",
            sum(len(v) for v in entities_a.values()), query_type,
        )
        _logger.info(
            "Retrieval: Tác vụ B → %d sub-queries, %d supplementary entities",
            len(decomposed), sum(len(v) for v in entities_b.values()),
        )
        _logger.info(
            "Retrieval: merged → %d entities (characters=%d, actors=%d, plays=%d, scenes=%d)",
            sum(len(v) for v in merged_entities.values()),
            len(merged_entities.get("characters", [])),
            len(merged_entities.get("actors", [])),
            len(merged_entities.get("plays", [])),
            len(merged_entities.get("scenes", [])),
        )

        processed: ProcessedQuery = {
            "original":   query,
            "expanded":   result_a["expanded"],
            "decomposed": decomposed,
        }

        return processed, merged_entities, query_type

    def _basic_pipeline(
        self, query: str
    ) -> tuple[ProcessedQuery, dict[str, list[str]], str]:
        """Pass-through query + 1 LLM call for entity extraction.

        Used by the 🧪 Experiment user-study page so GraphRAG and the vector
        RAG baseline are compared on equal terms (RAG has no query rewriting,
        so GraphRAG runs without it here too). NOT a deprecated fallback.

        Trade-offs vs the enhanced pipeline:
          - 1 LLM call (vs 2 parallel) → lower latency
          - Entity extraction NOT catalog-grounded → may hallucinate names
            that don't exist in the KG (those just yield 0 matches in Cypher)
          - No query classification → ``query_type`` is hard-defaulted to
            :attr:`QueryType.DEFAULT` ("Community") so auto-routing still
            picks a sensible methods/strategy set
        """
        _logger.info("Retrieval: enhancement disabled — processing query as-is")
        processed = self._query_processor.process(query, enable_enhancement=False)
        entities  = self._entity_extractor.extract(processed["expanded"])
        return processed, entities, QueryType.DEFAULT
