"""Graph retrieval from Neo4j across four strategies: nodes, triplets, paths, subgraph."""
from __future__ import annotations

from typing import Any, TypedDict

from src.constants.constant import EntityType, Limit, NodeProp, NodeType, RelType, RetrievalMethod
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.logger import get_logger

_logger = get_logger(__name__)

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


class GraphRetriever:
    """Retrieve graph data from Neo4j using a shared :class:`Neo4jClient`."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client
        _logger.info("GraphRetriever initialised")

    def retrieve(
        self,
        entities: dict[str, list[str]],
        methods: list[str] | None = None,
    ) -> GraphData:
        """Run one or more retrieval methods and return combined graph data."""
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
            except Exception as exc:
                _logger.error("Retrieval method '%s' failed: %s", method, exc)

        return graph_data

    def _get_nodes(self, entities: dict[str, list[str]]) -> list[NodeDict]:
        nodes: list[NodeDict] = []
        seen: set[str] = set()

        names = _all_names(entities)
        if not names:
            return nodes

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

        # Actor↔Play link via 5-hop walk: instance edges for `isPerformedIn`
        # aren't materialized, so we traverse Actor←RA→Version←Scene←Play.
        cypher = f"""
            MATCH (a:{NodeType.ACTOR})<-[:{RelType.PERFORMED_BY}]-(ra:{NodeType.ROLE_ASSIGNMENT})
                  -[:{RelType.IN_VERSION}]->(:{NodeType.VERSION})
                  <-[:{RelType.HAS_VERSION}]-(:{NodeType.SCENE})
                  <-[:{RelType.HAS_SCENE}]-(p:{NodeType.PLAY})
            WHERE any(n IN $names WHERE toLower(a.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n)
                                     OR toLower(p.{NodeProp.TITLE})      CONTAINS toLower(n))
            RETURN DISTINCT a.{NodeProp.ACTOR_NAME} AS subj, 'PERFORMS_IN' AS rel, p.{NodeProp.TITLE} AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen:
                seen.add(t)
                triplets.append(t)

        # coalesce keeps the triple informative across versionNumber / vidVersion / id.
        cypher = f"""
            MATCH (ra:{NodeType.ROLE_ASSIGNMENT})-[:{RelType.FOR_CHARACTER}]->(c:{NodeType.CHARACTER})
            MATCH (ra)-[:{RelType.PERFORMED_BY}]->(a:{NodeType.ACTOR})
            MATCH (ra)-[:{RelType.IN_VERSION}]->(v:{NodeType.VERSION})
            WHERE any(n IN $names WHERE toLower(c.{NodeProp.CHAR_NAME})  CONTAINS toLower(n)
                                     OR toLower(a.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n))
            RETURN (c.{NodeProp.CHAR_NAME} + ' (' + a.{NodeProp.ACTOR_NAME} + ')') AS subj,
                   '{RelType.IN_VERSION}' AS rel,
                   coalesce(v.{NodeProp.VERSION_NUMBER}, v.{NodeProp.VID_VERSION}, v.{NodeProp.ID}) AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen and t[0] and t[2]:
                seen.add(t)
                triplets.append(t)

        cypher = f"""
            MATCH (ra:{NodeType.ROLE_ASSIGNMENT})-[:{RelType.FOR_CHARACTER}]->(c:{NodeType.CHARACTER})
            MATCH (ra)-[:{RelType.HAS_APPEARANCE}]->(app:{NodeType.APPEARANCE})
            WHERE any(n IN $names WHERE toLower(c.{NodeProp.CHAR_NAME}) CONTAINS toLower(n))
              AND (app.{NodeProp.EMOTION} IS NOT NULL OR app.{NodeProp.SUBTITLE} IS NOT NULL)
            RETURN c.{NodeProp.CHAR_NAME} AS subj,
                   coalesce(app.{NodeProp.EMOTION}, 'lời thoại') AS rel,
                   coalesce(app.{NodeProp.SUBTITLE}, app.{NodeProp.EMOTION}, '') AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"] or "", r["obj"] or "")
            if t not in seen and t[0] and t[2]:
                seen.add(t)
                triplets.append(t)

        # Skip placeholder/Other mood values which carry no semantic content.
        cypher = f"""
            MATCH (a:{NodeType.ACTOR})-[:{RelType.EXPRESS}]->(m:{NodeType.MOOD})
            WHERE any(n IN $names WHERE toLower(a.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n))
              AND m.{NodeProp.EMOTION} IS NOT NULL
              AND m.{NodeProp.EMOTION} <> '...'
              AND m.{NodeProp.EMOTION} <> 'Other'
            RETURN DISTINCT a.{NodeProp.ACTOR_NAME} AS subj,
                            '{RelType.EXPRESS}' AS rel,
                            m.{NodeProp.EMOTION} AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen and t[0] and t[2]:
                seen.add(t)
                triplets.append(t)

        cypher = f"""
            MATCH (co:{NodeType.COSTUME})-[:{RelType.IS_WEAR_BY}]->(a:{NodeType.ACTOR})
            WHERE any(n IN $names WHERE toLower(a.{NodeProp.ACTOR_NAME}) CONTAINS toLower(n))
            RETURN DISTINCT co.{NodeProp.LABEL} AS subj,
                            '{RelType.IS_WEAR_BY}' AS rel,
                            a.{NodeProp.ACTOR_NAME} AS obj
            LIMIT $lim
        """
        for r in self._client.read(cypher, {"names": names, "lim": Limit.TRIPLET_QUERY}):
            t = (r["subj"] or "", r["rel"], r["obj"] or "")
            if t not in seen and t[0] and t[2]:
                seen.add(t)
                triplets.append(t)

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
            center = dict(record["center"])
            c_id = str(center)
            if c_id not in seen_nodes:
                seen_nodes.add(c_id)
                subgraph["nodes"].append(center)

            for node_list in record["all_nodes"]:
                for node in node_list:
                    if node is not None:
                        n = dict(node)
                        n_id = str(n)
                        if n_id not in seen_nodes:
                            seen_nodes.add(n_id)
                            subgraph["nodes"].append(n)

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
