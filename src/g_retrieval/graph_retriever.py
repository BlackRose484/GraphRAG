"""
Graph retrieval from Neo4j.

Provides five complementary retrieval strategies:
  - nodes      → individual matching nodes
  - triplets   → (subject, relationship, object) triples
  - paths      → multi-hop connection chains
  - subgraph   → ego-subgraph centred on matched entities
  - community  → pre-loaded play-centric community subgraphs

All Cypher queries are built using typed constants (NodeType, RelType, NodeProp,
Limit) so that changes to the ontology propagate automatically.

All entity names are batched into list parameters ($names) so each retrieval
method issues a fixed number of Cypher queries regardless of entity count:
  nodes     → 4 queries (one per label)
  triplets  → 3 queries (one per relationship pattern)
  paths     → 1 query
  subgraph  → 1 query
"""
from __future__ import annotations

from typing import Any, TypedDict

from src.constants.constant import EntityType, Limit, NodeProp, NodeType, RelType, RetrievalMethod
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

NodeDict = dict[str, Any]
Triplet = tuple[str, str, str]

_ALL_ENTITY_TYPES = EntityType.ALL
_PATH_ENTITY_TYPES = (EntityType.CHARACTERS, EntityType.ACTORS, EntityType.PLAYS)


class SubgraphResult(TypedDict):
    nodes: list[NodeDict]
    relationships: list[dict[str, Any]]


class GraphData(TypedDict):
    nodes: list[NodeDict]
    triplets: list[Triplet]
    paths: list[list[str]]
    subgraph: SubgraphResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_names(entities: dict[str, list[str]]) -> list[str]:
    names: list[str] = []
    for key in _ALL_ENTITY_TYPES:
        names.extend(entities.get(key, []))
    return names


def _path_names(entities: dict[str, list[str]]) -> list[str]:
    names: list[str] = []
    for key in _PATH_ENTITY_TYPES:
        names.extend(entities.get(key, []))
    return names


# ── Main class ────────────────────────────────────────────────────────────────


class GraphRetriever:
    """Retrieve graph data from Neo4j using a shared :class:`Neo4jClient`."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client
        _logger.info("GraphRetriever initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        entities: dict[str, list[str]],
        methods: list[str] | None = None,
        *,
        community_index: Any | None = None,
    ) -> GraphData:
        """Run one or more retrieval methods and return combined graph data.

        Args:
            entities: Extracted entities.
            methods: Any subset of ``RetrievalMethod.ALL``.
            community_index: Optional :class:`~community_index.CommunityIndex`
                             required when ``'community'`` is in *methods*.

        Returns:
            :class:`GraphData` dict with lists for every key.
        """
        if methods is None:
            methods = RetrievalMethod.DEFAULT

        graph_data: GraphData = {
            "nodes": [],
            "triplets": [],
            "paths": [],
            "subgraph": {"nodes": [], "relationships": []},
        }

        dispatch = {
            RetrievalMethod.NODES:    self._get_nodes,
            RetrievalMethod.TRIPLETS: self._get_triplets,
            RetrievalMethod.PATHS:    self._get_paths,
            RetrievalMethod.SUBGRAPH: self._get_subgraph,
        }

        for method in methods:
            # Community is handled separately (needs community_index)
            if method == RetrievalMethod.COMMUNITY:
                if community_index is None or not community_index.is_loaded():
                    _logger.debug("Community method skipped — no index loaded")
                    continue
                try:
                    comm_data = self._get_community(entities, community_index)
                    # Merge community nodes & triplets into graph_data
                    graph_data["nodes"].extend(comm_data.get("nodes", []))
                    graph_data["triplets"].extend(comm_data.get("triplets", []))
                    # Store raw community context for format_converter
                    graph_data["community_context"] = comm_data.get(  # type: ignore[typeddict-unknown-key]
                        "community_context", ""
                    )
                    n_nodes = len(comm_data.get("nodes", []))
                    n_trips = len(comm_data.get("triplets", []))
                    _logger.info(
                        "Method 'community' retrieved %d nodes, %d triplets",
                        n_nodes, n_trips,
                    )
                except Exception as exc:  # noqa: BLE001
                    _logger.error("Community retrieval failed: %s", exc)
                continue

            if method not in dispatch:
                _logger.warning("Unknown retrieval method: %s — skipped", method)
                continue
            try:
                result = dispatch[method](entities)
                if method == "subgraph":
                    graph_data["subgraph"] = result  # type: ignore[typeddict-item]
                else:
                    graph_data[method] = result  # type: ignore[literal-required]
                _logger.info(
                    "Method '%s' retrieved %s items",
                    method,
                    len(result) if isinstance(result, list) else len(result.get("nodes", [])),
                )
            except Exception as exc:  # noqa: BLE001
                _logger.error("Retrieval method '%s' failed: %s", method, exc)

        return graph_data

    # ── Retrieval strategies ──────────────────────────────────────────────────

    def _get_nodes(self, entities: dict[str, list[str]]) -> list[NodeDict]:
        nodes: list[NodeDict] = []
        seen: set[str] = set()

        names = _all_names(entities)
        if not names:
            return nodes

        # One query per label, all names batched — 4 queries instead of N×4
        label_queries: list[tuple[str, str, str]] = [
            (NodeType.CHARACTER, NodeProp.CHAR_NAME,  "c"),
            (NodeType.ACTOR,     NodeProp.ACTOR_NAME, "a"),
            (NodeType.PLAY,      NodeProp.TITLE,       "p"),
            (NodeType.SCENE,     NodeProp.SCENE_NAME,  "s"),
        ]

        for label, prop, alias in label_queries:
            cypher = (
                f"MATCH ({alias}:{label}) "
                f"WHERE any(n IN $names WHERE toLower({alias}.{prop}) CONTAINS toLower(n)) "
                f"RETURN {alias} LIMIT {Limit.NODE_QUERY}"
            )
            for record in self._client.read(cypher, {"names": names}):
                node = dict(record[alias])
                uid = str(node)
                if uid not in seen:
                    seen.add(uid)
                    nodes.append(node)

        # Broad fuzzy fallback when nothing matched — all terms in one query
        if not nodes and names:
            fallback_labels = " OR ".join(
                f"n:{t}" for t in NodeType.SEARCHABLE
            )
            terms = names[: Limit.FALLBACK_NAMES_COUNT]
            cypher = f"""
                MATCH (n)
                WHERE ({fallback_labels})
                  AND any(term IN $terms WHERE
                          any(p IN keys(n)
                              WHERE toLower(toString(n[p])) CONTAINS toLower(term)))
                RETURN n LIMIT $lim
            """
            for record in self._client.read(cypher, {"terms": terms, "lim": Limit.FALLBACK_NODES}):
                node = dict(record["n"])
                uid = str(node)
                if uid not in seen:
                    seen.add(uid)
                    nodes.append(node)

        return nodes

    def _get_triplets(self, entities: dict[str, list[str]]) -> list[Triplet]:
        triplets: list[Triplet] = []
        seen: set[Triplet] = set()

        names = _all_names(entities)
        if not names:
            return triplets

        # Character ↔ Actor (via RoleAssignment) — all names at once
        cypher = f"""
            MATCH (c:{NodeType.CHARACTER})-[:{RelType.FOR_CHARACTER}]-(ra:{NodeType.ROLE_ASSIGNMENT})-[:{RelType.PERFORMED_BY}]->(a:{NodeType.ACTOR})
            WHERE any(n IN $names WHERE toLower(c.{NodeProp.CHAR_NAME}) CONTAINS toLower(n)
                                     OR toLower(a.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n))
            RETURN c.{NodeProp.CHAR_NAME} AS subj, '{RelType.PERFORMED_BY}' AS rel, a.{NodeProp.ACTOR_NAME} AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t: Triplet = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen:
                seen.add(t)
                triplets.append(t)

        # Play → Character — all names at once
        cypher = f"""
            MATCH (p:{NodeType.PLAY})-[:{RelType.HAS_CHARACTER}]->(c:{NodeType.CHARACTER})
            WHERE any(n IN $names WHERE toLower(c.{NodeProp.CHAR_NAME}) CONTAINS toLower(n)
                                     OR toLower(p.{NodeProp.TITLE}) CONTAINS toLower(n))
            RETURN p.{NodeProp.TITLE} AS subj, '{RelType.HAS_CHARACTER}' AS rel, c.{NodeProp.CHAR_NAME} AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen:
                seen.add(t)
                triplets.append(t)

        # Play → Scene — all names at once
        cypher = f"""
            MATCH (p:{NodeType.PLAY})-[:{RelType.HAS_SCENE}]->(s:{NodeType.SCENE})
            WHERE any(n IN $names WHERE toLower(p.{NodeProp.TITLE}) CONTAINS toLower(n)
                                     OR toLower(s.{NodeProp.SCENE_NAME}) CONTAINS toLower(n))
            RETURN p.{NodeProp.TITLE} AS subj, '{RelType.HAS_SCENE}' AS rel, s.{NodeProp.SCENE_NAME} AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen:
                seen.add(t)
                triplets.append(t)

        # Fallback: return a sample of character–actor relationships
        if not triplets and names:
            cypher = f"""
                MATCH (c:{NodeType.CHARACTER})-[:{RelType.FOR_CHARACTER}]-(ra:{NodeType.ROLE_ASSIGNMENT})-[:{RelType.PERFORMED_BY}]->(a:{NodeType.ACTOR})
                RETURN c.{NodeProp.CHAR_NAME} AS subj, '{RelType.PERFORMED_BY}' AS rel, a.{NodeProp.ACTOR_NAME} AS obj
                LIMIT $lim
            """
            for r in self._client.read(cypher, {"lim": Limit.FALLBACK_TRIPLETS}):
                t = (r["subj"] or "", r["rel"], r["obj"] or "")
                if t not in seen:
                    seen.add(t)
                    triplets.append(t)

        return triplets

    def _get_paths(self, entities: dict[str, list[str]]) -> list[list[str]]:
        paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        names = _path_names(entities)
        if not names:
            return paths

        # All names batched into a single path query
        cypher = f"""
            MATCH path = (start)-[*1..{Limit.PATH_HOPS_MAX}]-(end)
            WHERE (
                (start:{NodeType.CHARACTER} AND any(n IN $names WHERE toLower(start.{NodeProp.CHAR_NAME})  CONTAINS toLower(n))) OR
                (start:{NodeType.ACTOR}     AND any(n IN $names WHERE toLower(start.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n))) OR
                (start:{NodeType.PLAY}      AND any(n IN $names WHERE toLower(start.{NodeProp.TITLE})      CONTAINS toLower(n)))
            ) AND (end:{NodeType.CHARACTER} OR end:{NodeType.ACTOR} OR end:{NodeType.PLAY} OR end:{NodeType.SCENE})
            RETURN [n IN nodes(path) |
                coalesce(n.{NodeProp.CHAR_NAME}, n.{NodeProp.ACTOR_NAME}, n.{NodeProp.TITLE}, n.{NodeProp.SCENE_NAME}, toString(id(n)))
            ] AS path_nodes
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.PATH_QUERY}):
            key = tuple(r["path_nodes"])
            if key not in seen:
                seen.add(key)
                paths.append(list(r["path_nodes"]))

        return paths

    def _get_subgraph(self, entities: dict[str, list[str]]) -> SubgraphResult:
        subgraph: SubgraphResult = {"nodes": [], "relationships": []}
        seen_nodes: set[str] = set()
        seen_rels: set[str] = set()

        names = _path_names(entities)
        if not names:
            return subgraph

        # All names batched into a single subgraph expansion query
        cypher = f"""
            MATCH path = (center)-[r*1..{Limit.SUBGRAPH_HOPS_MAX}]-(neighbor)
            WHERE (
                (center:{NodeType.CHARACTER} AND any(n IN $names WHERE toLower(center.{NodeProp.CHAR_NAME})  CONTAINS toLower(n))) OR
                (center:{NodeType.ACTOR}     AND any(n IN $names WHERE toLower(center.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n))) OR
                (center:{NodeType.PLAY}      AND any(n IN $names WHERE toLower(center.{NodeProp.TITLE})      CONTAINS toLower(n)))
            )
            WITH center, relationships(path) AS rels, nodes(path) AS path_nodes
            RETURN center,
                   collect(distinct path_nodes) AS all_nodes,
                   [rel IN rels | {{
                       type: type(rel),
                       start: startNode(rel),
                       end: endNode(rel)
                   }}] AS all_rels
            LIMIT $lim
        """
        for record in self._client.read(cypher, {"names": names, "lim": Limit.SUBGRAPH_QUERY}):
            # Center node
            center = dict(record["center"])
            c_id = str(center)
            if c_id not in seen_nodes:
                seen_nodes.add(c_id)
                subgraph["nodes"].append(center)

            # All nodes in paths
            for node_list in record["all_nodes"]:
                for node in node_list:
                    if node is not None:
                        n = dict(node)
                        n_id = str(n)
                        if n_id not in seen_nodes:
                            seen_nodes.add(n_id)
                            subgraph["nodes"].append(n)

            # Relationships
            for rel_info in record["all_rels"]:
                if not rel_info:
                    continue
                start = dict(rel_info.get("start") or {})
                end = dict(rel_info.get("end") or {})
                rel_id = f"{start}-{rel_info.get('type')}-{end}"
                if rel_id not in seen_rels:
                    seen_rels.add(rel_id)
                    subgraph["relationships"].append({
                        "type": rel_info.get("type", "UNKNOWN"),
                        "start": start,
                        "end": end,
                    })

        return subgraph

    # ── Community (pre-loaded) ────────────────────────────────────────────────

    @staticmethod
    def _get_community(
        entities: dict[str, list[str]],
        community_index: Any,
    ) -> dict[str, Any]:
        """Resolve entities to play communities and return merged graph data.

        Uses the pre-loaded :class:`~community_index.CommunityIndex` — no
        Cypher queries are executed at query time.

        Args:
            entities: Extracted entity dict.
            community_index: Loaded :class:`CommunityIndex` instance.

        Returns:
            Dict with ``nodes``, ``triplets``, ``community_context`` keys.
        """
        # Collect all entity names
        all_names: list[str] = []
        for key in _ALL_ENTITY_TYPES:
            all_names.extend(entities.get(key, []))

        if not all_names:
            return {"nodes": [], "triplets": [], "community_context": ""}

        # Resolve to communities
        communities = community_index.resolve_many(all_names)

        if not communities:
            _logger.debug("Community: no plays matched for entities %s", all_names)
            return {"nodes": [], "triplets": [], "community_context": ""}

        play_titles = [c.play_title for c in communities]
        _logger.info("Community: resolved %d plays: %s", len(play_titles), play_titles)

        return community_index.as_graph_data(play_titles)
