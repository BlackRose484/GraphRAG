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
    CHARACTER      = "Character"
    ACTOR          = "Actor"
    PLAY           = "Play"
    SCENE          = "Scene"
    VERSION        = "Version"
    ROLE_ASSIGNMENT = "RoleAssignment"
    APPEARANCE     = "Appearance"

    # Ordered list used when iterating all searchable node types
    ALL: list[str] = [
        CHARACTER, ACTOR, PLAY, SCENE,
        VERSION, ROLE_ASSIGNMENT, APPEARANCE,
    ]
    # Node types that appear in search results (main 4)
    SEARCHABLE: list[str] = [CHARACTER, ACTOR, PLAY, SCENE]


# ── Neo4j relationship types ──────────────────────────────────────────────────

class RelType:
    HAS_CHARACTER  = "HAS_CHARACTER"
    HAS_SCENE      = "HAS_SCENE"
    HAS_VERSION    = "HAS_VERSION"
    FOR_CHARACTER  = "FOR_CHARACTER"
    PERFORMED_BY   = "PERFORMED_BY"
    IN_VERSION     = "IN_VERSION"
    HAS_APPEARANCE = "HAS_APPEARANCE"

    ALL: list[str] = [
        HAS_CHARACTER, HAS_SCENE, HAS_VERSION,
        FOR_CHARACTER, PERFORMED_BY, IN_VERSION, HAS_APPEARANCE,
    ]

    # RDF object property name  →  (from_type, to_type, neo4j_rel)
    OWL_MAPPING: list[tuple] = [
        ("hasCharacter",  "Play",           "Character",  HAS_CHARACTER),
        ("hasScene",      "Play",           "Scene",      HAS_SCENE),
        ("hasVersion",    "Scene",          "Version",    HAS_VERSION),
        ("performedBy",   "RoleAssignment", "Actor",      PERFORMED_BY),
        ("forCharacter",  "RoleAssignment", "Character",  FOR_CHARACTER),
        ("inVersion",     "RoleAssignment", "Version",    IN_VERSION),
        ("hasAppearance", "RoleAssignment", "Appearance", HAS_APPEARANCE),
    ]


# ── Node property names ───────────────────────────────────────────────────────

class NodeProp:
    # Character
    CHAR_NAME   = "charName"
    CHAR_GENDER = "charGender"
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
    COMMUNITY = "community"

    ALL: list[str] = [NODES, TRIPLETS, PATHS, SUBGRAPH, COMMUNITY]
    DEFAULT: list[str] = [NODES, TRIPLETS, PATHS, SUBGRAPH, COMMUNITY]


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
