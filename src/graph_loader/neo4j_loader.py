"""Load Chèo ontology (Turtle/RDF) into Neo4j."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from collections import defaultdict

from rdflib import Graph, Namespace

from src.core.base import BaseLoader
from src.core.settings import settings
from src.constants import NodeType, RelType, NodeProp, ONTOLOGY_NAMESPACE
from src.constants.constant import CHARACTER_SUBCLASS_TO_ROLE
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadResult:
    nodes_created:         int = 0
    relationships_created: int = 0
    triples_parsed:        int = 0
    skipped_nodes:         int = 0
    errors:                list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        return (
            f"LoadResult("
            f"nodes={self.nodes_created}, "
            f"rels={self.relationships_created}, "
            f"triples={self.triples_parsed}, "
            f"skipped={self.skipped_nodes}, "
            f"errors={len(self.errors)})"
        )


class Neo4jLoader(BaseLoader):
    """Parse a Chèo Turtle ontology file and populate a Neo4j database."""

    _CONSTRAINTS = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Play)           REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Character)      REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Actor)          REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Scene)          REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Version)        REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RoleAssignment) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Appearance)     REQUIRE n.id IS UNIQUE",
    ]

    def __init__(self, client: Optional[Neo4jClient] = None) -> None:
        self._client       = client or Neo4jClient()
        self._owns_client  = client is None   # close only if we created it
        self._cheo_ns      = Namespace(ONTOLOGY_NAMESPACE)

    def __enter__(self) -> "Neo4jLoader":
        self._client.connect()
        return self

    def __exit__(self, *_) -> None:
        if self._owns_client:
            self._client.close()

    def load(self, source: Optional[str] = None) -> LoadResult:  # type: ignore[override]
        """Parse ``source`` (Turtle file path) and write to Neo4j."""
        ttl_path = Path(source) if source else settings.ontology.file_path

        if not ttl_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ttl_path}")

        logger.info("Loading ontology: %s", ttl_path)
        result = LoadResult()

        g = Graph()
        g.parse(str(ttl_path), format="turtle")
        result.triples_parsed = len(g)
        logger.info("Parsed %d RDF triples", result.triples_parsed)

        self._client.connect()

        self._create_constraints()
        result.nodes_created, result.skipped_nodes = self._load_individuals(g)
        result.relationships_created = self._load_relationships(g)

        logger.info("Load complete: %s", result)
        return result

    def clear(self) -> None:
        """Delete ALL nodes and relationships. Use with caution."""
        logger.warning("Clearing entire Neo4j database")
        self._client.write("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")

    def _create_constraints(self) -> None:
        logger.debug("Ensuring constraints exist")
        for cypher in self._CONSTRAINTS:
            self._client.write(cypher)
        logger.debug("Constraints OK")

    # Predicates excluded from the data-property pass — they are either
    # rdf/rdfs meta or object properties materialised as Neo4j relationships.
    _SKIP_DATATYPE_PREDICATES = frozenset({
        "type", "label", "subClassOf",
        "hasCharacter", "hasScene", "hasVersion",
        "performedBy", "forCharacter", "inVersion", "hasAppearance",
        "express", "isWearBy", "represent",
        "hasRelation", "collaboratesWith",
    })

    def _load_individuals(self, g: Graph) -> tuple[int, int]:
        """Return (created_count, skipped_count).

        Collect all rdf:types per individual first, then decide primary Neo4j
        label and secondary labels — each individual is MERGEd exactly once.
        """
        types_by_ind: dict = defaultdict(set)
        label_by_ind: dict = {}
        subject_by_ind: dict = {}

        sparql = """
        SELECT ?individual ?type ?label
        WHERE {
            ?individual rdf:type ?type .
            OPTIONAL { ?individual rdfs:label ?label }
            FILTER(?type != owl:NamedIndividual)
        }
        """
        for row in g.query(sparql):
            ind = row.individual
            type_name = str(row.type).split("#")[-1]
            types_by_ind[ind].add(type_name)
            subject_by_ind[ind] = ind
            if row.label is not None:
                label_by_ind[ind] = str(row.label)

        created = skipped = 0
        for ind, types in types_by_ind.items():
            primary, secondaries, role_type = self._classify_types(types)
            if primary is None:
                skipped += 1
                logger.debug(
                    "Skipped individual with no primary type: %s (types=%s)",
                    str(ind).split("#")[-1], sorted(types),
                )
                continue

            individual_id = str(ind).split("#")[-1]
            label = label_by_ind.get(ind, individual_id)

            props = self._extract_data_properties(g, subject_by_ind[ind])
            props[NodeProp.ID]    = individual_id
            props[NodeProp.LABEL] = label
            if role_type:
                props[NodeProp.ROLE_TYPE] = role_type

            self._merge_node(primary, secondaries, props)
            created += 1

        logger.info("Nodes: %d created, %d skipped", created, skipped)
        return created, skipped

    @staticmethod
    def _classify_types(
        types: set,
    ) -> tuple[str | None, list[str], str | None]:
        primary: str | None = None
        for t in types:
            if t in NodeType.PRIMARY:
                primary = t
                break

        secondaries: list[str] = []
        if primary == NodeType.APPEARANCE:
            secondaries.extend(
                t for t in types if t in NodeType.APPEARANCE_SUBTYPES
            )
        # Validation tags from inference rules apply to any primary type.
        secondaries.extend(
            t for t in types if t in NodeType.VALIDATION_TAGS
        )
        secondaries = sorted(set(secondaries))

        role_type: str | None = None
        if primary == NodeType.CHARACTER:
            for t in types:
                if t in CHARACTER_SUBCLASS_TO_ROLE:
                    role_type = CHARACTER_SUBCLASS_TO_ROLE[t]
                    break

        return primary, secondaries, role_type

    def _extract_data_properties(self, g: Graph, subject) -> Dict[str, str]:
        props: Dict[str, str] = {}
        for predicate, obj in g.predicate_objects(subject):
            pred_name = str(predicate).split("#")[-1]
            if pred_name in self._SKIP_DATATYPE_PREDICATES:
                continue
            props[pred_name] = str(obj)
        return props

    def _merge_node(
        self,
        primary: str,
        secondaries: list[str],
        props: Dict[str, str],
    ) -> None:
        """MERGE node by id with primary + optional secondary labels.

        Secondary labels are added via SET because Cypher MERGE only matches
        the primary label — putting more labels in the pattern would risk
        creating duplicates.
        """
        set_assignments = [f"n.{k} = ${k}" for k in props if k != NodeProp.ID]
        for sec in secondaries:
            set_assignments.append(f"n:{sec}")
        set_clause = ("SET " + ", ".join(set_assignments)) if set_assignments else ""
        cypher = f"MERGE (n:{primary} {{id: $id}}) {set_clause}".strip()
        self._client.write(cypher, props)

    def _load_relationships(self, g: Graph) -> int:
        total = 0

        for rdf_prop, from_type, to_type, neo4j_rel in RelType.OWL_MAPPING:
            predicate_uri = str(self._cheo_ns[rdf_prop])
            sparql = f"""
            SELECT ?from ?to
            WHERE {{
                ?from <{predicate_uri}> ?to .
            }}
            """
            count = 0
            for row in g.query(sparql):
                from_id = str(row["from"]).split("#")[-1]
                to_id   = str(row["to"]).split("#")[-1]
                self._merge_relationship(from_type, from_id, to_type, to_id, neo4j_rel)
                count += 1

            logger.debug(
                "Relationship %s → %s [%s]: %d", from_type, to_type, neo4j_rel, count
            )
            total += count

        logger.info("Relationships: %d created", total)
        return total

    def _merge_relationship(
        self,
        from_type: str, from_id: str,
        to_type:   str, to_id:   str,
        rel_type:  str,
    ) -> None:
        cypher = f"""
        MATCH (a:{from_type} {{id: $from_id}})
        MATCH (b:{to_type}   {{id: $to_id}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        self._client.write(cypher, {"from_id": from_id, "to_id": to_id})
