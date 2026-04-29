"""
Domain constants for the Chèo GraphRAG system.

All magic strings, node/relationship type names, property keys, method names,
and numeric defaults live here. Nothing else in the codebase should hardcode
these values directly.
"""

from __future__ import annotations


# ── Ontology ──────────────────────────────────────────────────────────────────

ONTOLOGY_NAMESPACE = "http://www.semanticweb.org/asus/ontologies/2025/5/Cheo#"


# ── Neo4j node labels ─────────────────────────────────────────────────────────

class NodeType:
    # --- Primary labels (one per individual) ---
    CHARACTER       = "Character"
    ACTOR           = "Actor"
    PLAY            = "Play"
    SCENE           = "Scene"
    VERSION         = "Version"
    ROLE_ASSIGNMENT = "RoleAssignment"
    APPEARANCE      = "Appearance"

    # --- Appearance subtype secondary labels (multi-label on top of Appearance) ---
    MOOD         = "Mood"
    COSTUME      = "Costume"
    FACE_GESTURE = "FaceGesture"

    # --- Reserved (schema only, no instances yet) ---
    ARTIFACT          = "Artifact"
    PHYSICAL_GESTURE  = "PhysicalGesture"
    LEG_GESTURE       = "LegGesture"
    HAND_GESTURE      = "HandGesture"
    COSTUME_ARTIFACT  = "CostumeArtifact"

    # --- Validation tags from inference rules (Phụ lục B) ---
    CROSS_GENDER_CASTING    = "CrossGenderCasting"
    INCOMPLETE_CHARACTER    = "IncompleteCharacter"
    ATYPICAL_ROLE           = "AtypicalRole"
    ORPHANED_APPEARANCE     = "OrphanedAppearance"

    PRIMARY: list[str] = [
        CHARACTER, ACTOR, PLAY, SCENE,
        VERSION, ROLE_ASSIGNMENT, APPEARANCE,
    ]
    APPEARANCE_SUBTYPES: list[str] = [MOOD, COSTUME, FACE_GESTURE]
    VALIDATION_TAGS: list[str] = [
        CROSS_GENDER_CASTING, INCOMPLETE_CHARACTER,
        ATYPICAL_ROLE, ORPHANED_APPEARANCE,
    ]

    # All Neo4j labels that may appear on a node (primary + subtype labels).
    ALL: list[str] = PRIMARY + APPEARANCE_SUBTYPES

    # Node types that appear in search results (main 4)
    SEARCHABLE: list[str] = [CHARACTER, ACTOR, PLAY, SCENE]


# Vietnamese mainType label assigned as a property to Character nodes whose
# RDF type is one of the Character subclasses (DaoCharacter, HeCharacter, ...).
CHARACTER_SUBCLASS_TO_ROLE: dict[str, str] = {
    "DaoCharacter": "Đào",
    "HeCharacter":  "Hề",
    "KepCharacter": "Kép",
    "MuCharacter":  "Mụ",
    "LaoCharacter": "Lão",
}


# ── Neo4j relationship types ──────────────────────────────────────────────────

class RelType:
    # --- Existing (v1 schema) ---
    HAS_CHARACTER  = "HAS_CHARACTER"
    HAS_SCENE      = "HAS_SCENE"
    HAS_VERSION    = "HAS_VERSION"
    FOR_CHARACTER  = "FOR_CHARACTER"
    PERFORMED_BY   = "PERFORMED_BY"
    IN_VERSION     = "IN_VERSION"
    HAS_APPEARANCE = "HAS_APPEARANCE"

    # --- New (v3 schema extension) ---
    EXPRESS            = "EXPRESS"             # Actor → Mood (data present)
    IS_WEAR_BY         = "IS_WEAR_BY"          # Costume → Actor (data present)
    IS_ACCOMPANIED_BY  = "IS_ACCOMPANIED_BY"   # Artifact → Actor (reserved)
    REPRESENT          = "REPRESENT"           # CostumeArtifact → Mood (reserved)
    FOLLOW             = "FOLLOW"              # Scene → Scene (reserved)
    HAS_RELATION       = "HAS_RELATION"        # Character → Character (reserved)
    IS_OPPONENT_OF     = "IS_OPPONENT_OF"      # Character → Character (reserved)
    TRAINED_BY         = "TRAINED_BY"          # Actor → Actor (reserved)
    COLLABORATES_WITH  = "COLLABORATES_WITH"   # Actor → Actor (data after B.3)

    # --- Inferred relations from Phụ lục B ---
    PERFORMS           = "PERFORMS"            # Actor → Character (B.2)
    ARCHETYPE_SIMILAR  = "ARCHETYPE_SIMILAR"   # Character → Character (C.3)

    ALL: list[str] = [
        HAS_CHARACTER, HAS_SCENE, HAS_VERSION,
        FOR_CHARACTER, PERFORMED_BY, IN_VERSION, HAS_APPEARANCE,
        EXPRESS, IS_WEAR_BY, IS_ACCOMPANIED_BY, REPRESENT,
        FOLLOW, HAS_RELATION, IS_OPPONENT_OF, TRAINED_BY, COLLABORATES_WITH,
        PERFORMS, ARCHETYPE_SIMILAR,
    ]

    # RDF object property name  →  (from_type, to_type, neo4j_rel)
    # `from_type` / `to_type` is the Cypher label used for MATCH; for Appearance
    # subtype targets we use the subtype label so Cypher binds the right node.
    OWL_MAPPING: list[tuple] = [
        ("hasCharacter",      "Play",           "Character",      HAS_CHARACTER),
        ("hasScene",          "Play",           "Scene",          HAS_SCENE),
        ("hasVersion",        "Scene",          "Version",        HAS_VERSION),
        ("performedBy",       "RoleAssignment", "Actor",          PERFORMED_BY),
        ("forCharacter",      "RoleAssignment", "Character",      FOR_CHARACTER),
        ("inVersion",         "RoleAssignment", "Version",        IN_VERSION),
        ("hasAppearance",     "RoleAssignment", "Appearance",     HAS_APPEARANCE),
        ("express",           "Actor",          "Mood",           EXPRESS),
        ("isWearBy",          "Costume",        "Actor",          IS_WEAR_BY),
        ("isAccompaniedBy",   "Artifact",       "Actor",          IS_ACCOMPANIED_BY),
        ("represent",         "Appearance",     "Mood",           REPRESENT),
        ("follow",            "Scene",          "Scene",          FOLLOW),
        ("hasRelation",       "Character",      "Character",      HAS_RELATION),
        ("isOpponentOf",      "Character",      "Character",      IS_OPPONENT_OF),
        ("trainedBy",         "Actor",          "Actor",          TRAINED_BY),
        ("collaboratesWith",  "Actor",          "Actor",          COLLABORATES_WITH),
        # v4 — derived from Phụ lục B
        ("performs",          "Actor",          "Character",      PERFORMS),
        ("archetypeSimilar",  "Character",      "Character",      ARCHETYPE_SIMILAR),
    ]


# ── Node property names ───────────────────────────────────────────────────────

class NodeProp:
    # Character
    CHAR_NAME   = "charName"
    CHAR_GENDER = "charGender"
    ROLE_TYPE   = "roleType"     # v3: derived from Character subclass (Đào / Hề / ...)
    SUB_TYPE    = "subType"      # v1: subtype string (Chín / Pha / Áo dài / ...)
    DESCRIPTION = "description"
    # Actor
    ACTOR_NAME  = "actorName"
    # Play
    TITLE       = "title"
    # Scene
    SCENE_NAME    = "sceneName"
    SCENE_SUMMARY = "sceneSummary"
    # Version
    VERSION_NUMBER = "versionNumber"
    VID_VERSION    = "vidVersion"
    # Appearance
    EMOTION     = "emotion"
    SUBTITLE    = "subtitle"
    START_TIME  = "startTime"
    END_TIME    = "endTime"
    # v3 — rdfs:comment surfaces here for Costume/FaceGesture nodes
    COMMENT     = "comment"
    # Common
    ID    = "id"
    LABEL = "label"

    # Display name lookup: property key → human label (Vietnamese)
    DISPLAY: dict[str, str] = {
        CHAR_NAME:    "Nhân vật",
        ACTOR_NAME:   "Diễn viên",
        TITLE:        "Vở kịch",
        SCENE_NAME:   "Trích đoạn",
    }


# ── Entity categories (used in extraction & retrieval) ────────────────────────

class EntityType:
    CHARACTERS = "characters"
    ACTORS     = "actors"
    PLAYS      = "plays"
    SCENES     = "scenes"

    ALL: list[str] = [CHARACTERS, ACTORS, PLAYS, SCENES]


# ── G-Retrieval methods ───────────────────────────────────────────────────────

class RetrievalMethod:
    NODES     = "nodes"
    TRIPLETS  = "triplets"
    PATHS     = "paths"
    SUBGRAPH  = "subgraph"
    # Internal-only: community-level cache used as global fallback,
    # not exposed as a primary retrieval method.
    COMMUNITY = "community"

    ALL: list[str] = [NODES, TRIPLETS, PATHS, SUBGRAPH]
    DEFAULT: list[str] = [NODES, TRIPLETS, PATHS, SUBGRAPH]


# ── Graph format converter keys ───────────────────────────────────────────────

class FormatKey:
    NATURAL_LANGUAGE  = "natural_language"
    ADJACENCY_TABLE   = "adjacency_table"
    CODE_LIKE         = "code_like"
    NODE_SEQUENCE     = "node_sequence"
    EMBEDDING_TEXT    = "embedding_text"
    COMMUNITY_SUMMARY = "community_summary"

    ALL: list[str] = [
        NATURAL_LANGUAGE, ADJACENCY_TABLE,
        CODE_LIKE, NODE_SEQUENCE, EMBEDDING_TEXT,
        COMMUNITY_SUMMARY,
    ]

    # Human-readable titles (Vietnamese) used in context building
    TITLES: dict[str, str] = {
        NATURAL_LANGUAGE:  "Mô tả tự nhiên",
        ADJACENCY_TABLE:   "Bảng quan hệ",
        CODE_LIKE:         "Chi tiết dữ liệu",
        NODE_SEQUENCE:     "Đường đi",
        EMBEDDING_TEXT:    "Text representation",
        COMMUNITY_SUMMARY: "Community subgraph",
    }

    # Default formats per generation strategy
    DEFAULT_PRE:  list[str] = [NATURAL_LANGUAGE, ADJACENCY_TABLE]
    DEFAULT_MID:  list[str] = [NATURAL_LANGUAGE, CODE_LIKE]
    DEFAULT_POST: list[str] = [NATURAL_LANGUAGE, ADJACENCY_TABLE, CODE_LIKE]


# ── G-Generation strategy names ───────────────────────────────────────────────

class GenerationStrategy:
    PRE  = "pre"
    MID  = "mid"
    POST = "post"

    ALL: list[str] = [PRE, MID, POST]
    DEFAULT: str   = MID  # mid provides the best structured guidance


# ── Query type for auto-routing ───────────────────────────────────────────────

class QueryType:
    """Classification of a user query, produced by the LLM in Tác vụ A.

    Used to auto-select retrieval methods and generation strategy when the
    pipeline runs with ``auto_routing=True``. When the LLM returns an
    unrecognised value the system falls back to :attr:`DEFAULT`.
    """

    LOCAL     = "Local"      # direct lookup of a single entity (1-2 hops)
    COMMUNITY = "Community"  # aggregation over a connected entity cluster
    GLOBAL    = "Global"     # cross-play synthesis or full-KG comparison

    ALL: list[str] = [LOCAL, COMMUNITY, GLOBAL]
    DEFAULT: str   = COMMUNITY  # safe middle when classification fails

    # Auto-routing: query type → activated retrieval methods
    METHODS: dict[str, list[str]] = {
        LOCAL:     [RetrievalMethod.NODES, RetrievalMethod.TRIPLETS],
        COMMUNITY: [RetrievalMethod.NODES, RetrievalMethod.TRIPLETS,
                    RetrievalMethod.PATHS],
        GLOBAL:    [RetrievalMethod.NODES, RetrievalMethod.TRIPLETS,
                    RetrievalMethod.PATHS, RetrievalMethod.SUBGRAPH],
    }

    # Auto-routing: query type → generation strategy
    STRATEGY: dict[str, str] = {
        LOCAL:     GenerationStrategy.PRE,
        COMMUNITY: GenerationStrategy.MID,
        GLOBAL:    GenerationStrategy.POST,
    }


# ── Retrieval / generation numeric limits ─────────────────────────────────────

class Limit:
    # Cypher query result caps
    NODE_SEARCH_PER_NAME  = 50   # max nodes returned per entity name
    FALLBACK_NODES        = 10   # fallback broad search limit
    FALLBACK_TRIPLETS     = 50   # fallback triplet limit
    PATH_HOPS_MAX         = 3    # max relationship hops for path queries
    SUBGRAPH_HOPS_MAX     = 2    # max hops for subgraph neighborhood

    # Per-method retrieval result caps (used in graph_retriever)
    NODE_QUERY     = 50  # max nodes returned per name per label (mirrors NODE_SEARCH_PER_NAME)
    TRIPLET_QUERY  = 50   # max triplets per name per relationship pattern
    PATH_QUERY     = 30   # max paths per name
    SUBGRAPH_QUERY = 10   # max subgraph expansions per name

    # Query processing
    FALLBACK_NAMES_COUNT  = 3    # how many names to use in broad fallback search

    # Format conversion
    SCENE_SUMMARY_MAX_LEN = 200  # chars before truncating sceneSummary
