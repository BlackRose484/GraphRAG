"""
Load Chèo ontology (Turtle/RDF) into Neo4j.

Improvements over GraphRAG v1:
- Extends BaseLoader ABC
- Zero magic strings — all types/properties from src.constants
- Proper logging (no print)
- Idempotent MERGE instead of CREATE → safe to re-run
- Progress counters returned as LoadResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from collections import defaultdict

from rdflib import Graph, Namespace, RDF, RDFS

from src.core.base import BaseLoader, ProcessingResult
from src.core.settings import settings
from src.constants import NodeType, RelType, NodeProp, ONTOLOGY_NAMESPACE
from src.constants.constant import CHARACTER_SUBCLASS_TO_ROLE
from src.graph_loader.neo4j_client import Neo4jClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

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


# ── Loader ────────────────────────────────────────────────────────────────────

class Neo4jLoader(BaseLoader):
    """
    Parse a Chèo Turtle ontology file and populate a Neo4j database.

    Usage::

        loader = Neo4jLoader()
        result = loader.load()          # uses path from settings
        result = loader.load("my.ttl")  # explicit path

        # Or use as context manager (auto-closes Neo4j driver):
        with Neo4jLoader() as loader:
            result = loader.load()
    """

    # Cypher constraints — one per unique-id node type
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

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Neo4jLoader":
        self._client.connect()
        return self

    def __exit__(self, *_) -> None:
        if self._owns_client:
            self._client.close()

    # ── BaseLoader ABC ────────────────────────────────────────────────────────

    def load(self, source: Optional[str] = None) -> LoadResult:  # type: ignore[override]
        """
        Parse ``source`` (Turtle file path) and write to Neo4j.

        Args:
            source: Path to .ttl file.  Defaults to ``settings.ontology.file_path``.

        Returns:
            :class:`LoadResult` with counts and any errors.
        """
        ttl_path = Path(source) if source else settings.ontology.file_path

        if not ttl_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ttl_path}")

        logger.info("Loading ontology: %s", ttl_path)
        result = LoadResult()

        # ── Parse RDF ─────────────────────────────────────────────────────────
        g = Graph()
        g.parse(str(ttl_path), format="turtle")
        result.triples_parsed = len(g)
        logger.info("Parsed %d RDF triples", result.triples_parsed)

        self._client.connect()

        # ── Setup schema ──────────────────────────────────────────────────────
        self._create_constraints()

        # ── Load nodes ────────────────────────────────────────────────────────
        result.nodes_created, result.skipped_nodes = self._load_individuals(g)

        # ── Load relationships ────────────────────────────────────────────────
        result.relationships_created = self._load_relationships(g)

        logger.info("Load complete: %s", result)
        return result

    def clear(self) -> None:
        """Delete ALL nodes and relationships. Use with caution."""
        logger.warning("Clearing entire Neo4j database")
        self._client.write("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")

    # ── Schema setup ─────────────────────────────────────────────────────────

    def _create_constraints(self) -> None:
        logger.debug("Ensuring constraints exist")
        for cypher in self._CONSTRAINTS:
            self._client.write(cypher)
        logger.debug("Constraints OK")

    # ── Individuals → nodes ───────────────────────────────────────────────────

    # Predicates excluded from the data-property pass — they are either
    # rdf/rdfs meta or object properties materialised as Neo4j relationships.
    _SKIP_DATATYPE_PREDICATES = frozenset({
        # rdf / rdfs / owl meta
        "type", "label", "subClassOf",
        # Existing object properties (v1)
        "hasCharacter", "hasScene", "hasVersion",
        "performedBy", "forCharacter", "inVersion", "hasAppearance",
        # New object properties (v3)
        "express", "isWearBy", "isAccompaniedBy", "represent",
        "follow", "hasRelation", "isOpponentOf",
        "trainedBy", "collaboratesWith",
    })

    def _load_individuals(self, g: Graph) -> tuple[int, int]:
        """Return (created_count, skipped_count).

        Strategy: collect *all* rdf:types per individual first, then decide
        a primary Neo4j label and any secondary labels / property-encoded
        subclass info. Each individual is MERGEd exactly once.
        """
        # individual_uri -> set of type local names
        types_by_ind: dict = defaultdict(set)
        # individual_uri -> rdfs:label (last one wins; usually unique)
        label_by_ind: dict = {}
        # individual_uri -> the rdflib subject node (needed for property extraction)
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
        """Pick (primary_label, secondary_labels, role_type) from rdf:types.

        - primary  = the matching `NodeType.PRIMARY` label.
        - secondaries = matching `NodeType.APPEARANCE_SUBTYPES` (only when
          primary is Appearance) PLUS any matching `NodeType.VALIDATION_TAGS`
          (regardless of primary).
        - role_type = Vietnamese mainType when one of the Character subclasses
          is present.
        """
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
        """Return datatype properties (non-object-properties) for a subject.

        Skips predicates listed in :attr:`_SKIP_DATATYPE_PREDICATES` so that
        object properties (turned into Neo4j relationships elsewhere) and
        meta predicates do not leak into node properties.
        """
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

        Idempotent — safe to re-run. The Cypher MERGE is on `(n:Primary {id})`,
        then secondary labels (if any) are added via SET because labels are
        not allowed in the MERGE pattern beyond the matched one without
        risking creating duplicates.
        """
        set_assignments = [f"n.{k} = ${k}" for k in props if k != NodeProp.ID]
        for sec in secondaries:
            set_assignments.append(f"n:{sec}")
        set_clause = ("SET " + ", ".join(set_assignments)) if set_assignments else ""
        cypher = f"MERGE (n:{primary} {{id: $id}}) {set_clause}".strip()
        self._client.write(cypher, props)

    # ── Object properties → relationships ─────────────────────────────────────

    def _load_relationships(self, g: Graph) -> int:
        """Return total relationships created."""
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
        """MERGE relationship — idempotent."""
        cypher = f"""
        MATCH (a:{from_type} {{id: $from_id}})
        MATCH (b:{to_type}   {{id: $to_id}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        self._client.write(cypher, {"from_id": from_id, "to_id": to_id})
