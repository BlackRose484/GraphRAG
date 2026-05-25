"""G-Retrieval orchestrator: ENHANCED (parallel A║B) and BASIC (single extract) paths."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any

from src.constants.constant import EntityType, FormatKey, QueryType, RetrievalMethod
from src.g_retrieval.entity_catalog import EntityCatalog
from src.g_retrieval.entity_extractor import EntityExtractor
from src.g_retrieval.graph_retriever import GraphData, GraphRetriever
from src.g_retrieval.query_processor import ProcessedQuery, QueryProcessor
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.format_converter import GraphFormatConverter
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_DEFAULT_METHODS: list[str] = RetrievalMethod.DEFAULT
_DEFAULT_FORMATS: list[str] = FormatKey.DEFAULT_MID


def _merge_entities(
    base: dict[str, list[str]],
    extra: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Case-insensitive union of two entity dicts."""
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
    query: str
    processed_query: ProcessedQuery
    graph_data: GraphData
    retrieval_methods: list[str]
    format_keys: list[str]
    formatted_contexts: dict[str, str]
    key_facts: str
    entities: dict[str, list[str]]
    query_type: str
    num_nodes: int
    num_triplets: int
    num_paths: int
    retrieval_time: float
    error: str | None = None

    def summary(self) -> dict[str, Any]:
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
    def __init__(self, client: Neo4jClient) -> None:
        self._entity_catalog   = EntityCatalog()
        self._entity_catalog.load(client)
        self._query_processor  = QueryProcessor()
        self._entity_extractor = EntityExtractor()
        self._graph_retriever  = GraphRetriever(client)
        _logger.info("RetrievalOrchestrator initialised")

    def retrieve(
        self,
        query: str,
        *,
        retrieval_methods: list[str] | None = None,
        format_keys: list[str] | None = None,
        enable_enhancement: bool = True,
        auto_routing: bool = True,
    ) -> RetrievalResult:
        explicit_methods = retrieval_methods is not None
        format_keys      = format_keys or _DEFAULT_FORMATS
        start = time.perf_counter()

        try:
            if enable_enhancement:
                processed, entities, query_type = self._enhanced_pipeline(query)
            else:
                processed, entities, query_type = self._basic_pipeline(query)

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

            _logger.info("Retrieval: querying Neo4j with methods=%s", retrieval_methods)
            graph_data = self._graph_retriever.retrieve(
                entities,
                methods=retrieval_methods,
            )

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

        except Exception as exc:
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

    def _enhanced_pipeline(
        self, query: str
    ) -> tuple[ProcessedQuery, dict[str, list[str]], str]:
        """Run Tiến trình A (expand+extract+classify) and B (decompose+extract) in parallel."""
        _logger.info("Retrieval: launching parallel Tiến trình A ║ Tiến trình B")

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
            result_a = future_a.result()
            result_b = future_b.result()

        entities_a:  dict[str, list[str]] = dict(result_a["entities"])
        entities_b:  dict[str, list[str]] = dict(result_b["entities"])
        decomposed:  list[str]            = list(result_b["decomposed"])
        query_type:  str                  = result_a["query_type"]

        merged_entities = _merge_entities(entities_a, entities_b)

        _logger.info(
            "Retrieval: Tiến trình A → %d entities, query_type=%s",
            sum(len(v) for v in entities_a.values()), query_type,
        )
        _logger.info(
            "Retrieval: Tiến trình B → %d sub-queries, %d supplementary entities",
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
        """Pass-through query + 1 LLM call (no catalog grounding, no classification)."""
        _logger.info("Retrieval: enhancement disabled — processing query as-is")
        processed = self._query_processor.process(query, enable_enhancement=False)
        entities  = self._entity_extractor.extract(processed["expanded"])
        return processed, entities, QueryType.DEFAULT
