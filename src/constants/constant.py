"""Domain constants for the Chèo GraphRAG system."""

from __future__ import annotations


ONTOLOGY_NAMESPACE = "http://www.semanticweb.org/asus/ontologies/2025/5/Cheo#"


class NodeType:
    CHARACTER       = "Character"
    ACTOR           = "Actor"
    PLAY            = "Play"
    SCENE           = "Scene"
    VERSION         = "Version"
    ROLE_ASSIGNMENT = "RoleAssignment"
    APPEARANCE      = "Appearance"

    MOOD         = "Mood"
    COSTUME      = "Costume"
    FACE_GESTURE = "FaceGesture"

    INCOMPLETE_CHARACTER = "IncompleteCharacter"
    ORPHANED_APPEARANCE  = "OrphanedAppearance"

    PRIMARY: list[str] = [
        CHARACTER, ACTOR, PLAY, SCENE,
        VERSION, ROLE_ASSIGNMENT, APPEARANCE,
    ]
    APPEARANCE_SUBTYPES: list[str] = [MOOD, COSTUME, FACE_GESTURE]
    VALIDATION_TAGS: list[str] = [INCOMPLETE_CHARACTER, ORPHANED_APPEARANCE]

    ALL: list[str] = PRIMARY + APPEARANCE_SUBTYPES
    SEARCHABLE: list[str] = [CHARACTER, ACTOR, PLAY, SCENE]


CHARACTER_SUBCLASS_TO_ROLE: dict[str, str] = {
    "DaoCharacter": "Đào",
    "HeCharacter":  "Hề",
    "KepCharacter": "Kép",
    "MuCharacter":  "Mụ",
    "LaoCharacter": "Lão",
}


class RelType:
    HAS_CHARACTER  = "HAS_CHARACTER"
    HAS_SCENE      = "HAS_SCENE"
    HAS_VERSION    = "HAS_VERSION"
    FOR_CHARACTER  = "FOR_CHARACTER"
    PERFORMED_BY   = "PERFORMED_BY"
    IN_VERSION     = "IN_VERSION"
    HAS_APPEARANCE = "HAS_APPEARANCE"

    EXPRESS            = "EXPRESS"
    IS_WEAR_BY         = "IS_WEAR_BY"
    REPRESENT          = "REPRESENT"
    HAS_RELATION       = "HAS_RELATION"
    COLLABORATES_WITH  = "COLLABORATES_WITH"

    PERFORMS           = "PERFORMS"

    ALL: list[str] = [
        HAS_CHARACTER, HAS_SCENE, HAS_VERSION,
        FOR_CHARACTER, PERFORMED_BY, IN_VERSION, HAS_APPEARANCE,
        EXPRESS, IS_WEAR_BY, REPRESENT,
        HAS_RELATION, COLLABORATES_WITH,
        PERFORMS,
    ]

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
        ("represent",         "Costume",        "Mood",           REPRESENT),
        ("hasRelation",       "Character",      "Character",      HAS_RELATION),
        ("collaboratesWith",  "Actor",          "Actor",          COLLABORATES_WITH),
        ("performs",          "Actor",          "Character",      PERFORMS),
    ]


class NodeProp:
    CHAR_NAME   = "charName"
    CHAR_GENDER = "charGender"
    ROLE_TYPE   = "roleType"
    SUB_TYPE    = "subType"
    DESCRIPTION = "description"
    ACTOR_NAME  = "actorName"
    TITLE       = "title"
    SCENE_NAME    = "sceneName"
    SCENE_SUMMARY = "sceneSummary"
    VERSION_NUMBER = "versionNumber"
    VID_VERSION    = "vidVersion"
    EMOTION     = "emotion"
    SUBTITLE    = "subtitle"
    START_TIME  = "startTime"
    END_TIME    = "endTime"
    COMMENT     = "comment"
    ID    = "id"
    LABEL = "label"

    DISPLAY: dict[str, str] = {
        CHAR_NAME:    "Nhân vật",
        ACTOR_NAME:   "Diễn viên",
        TITLE:        "Vở kịch",
        SCENE_NAME:   "Trích đoạn",
    }


class EntityType:
    CHARACTERS = "characters"
    ACTORS     = "actors"
    PLAYS      = "plays"
    SCENES     = "scenes"

    ALL: list[str] = [CHARACTERS, ACTORS, PLAYS, SCENES]


class RetrievalMethod:
    NODES     = "nodes"
    TRIPLETS  = "triplets"
    PATHS     = "paths"
    SUBGRAPH  = "subgraph"

    ALL: list[str] = [NODES, TRIPLETS, PATHS, SUBGRAPH]
    DEFAULT: list[str] = [NODES, TRIPLETS, PATHS, SUBGRAPH]


class FormatKey:
    NATURAL_LANGUAGE  = "natural_language"
    ADJACENCY_TABLE   = "adjacency_table"
    CODE_LIKE         = "code_like"
    NODE_SEQUENCE     = "node_sequence"

    ALL: list[str] = [
        NATURAL_LANGUAGE, ADJACENCY_TABLE,
        CODE_LIKE, NODE_SEQUENCE,
    ]

    TITLES: dict[str, str] = {
        NATURAL_LANGUAGE:  "Mô tả tự nhiên",
        ADJACENCY_TABLE:   "Bảng quan hệ",
        CODE_LIKE:         "Chi tiết dữ liệu",
        NODE_SEQUENCE:     "Đường đi",
    }

    DEFAULT_PRE:  list[str] = [NATURAL_LANGUAGE, ADJACENCY_TABLE]
    DEFAULT_MID:  list[str] = [NATURAL_LANGUAGE, ADJACENCY_TABLE, CODE_LIKE]
    DEFAULT_POST: list[str] = [NATURAL_LANGUAGE, ADJACENCY_TABLE, CODE_LIKE]


class GenerationStrategy:
    PRE  = "pre"
    MID  = "mid"
    POST = "post"

    ALL: list[str] = [PRE, MID, POST]
    DEFAULT: str   = MID


class QueryType:
    """Query classification produced by the LLM in Tiến trình A."""

    LOCAL     = "Local"
    COMMUNITY = "Community"
    GLOBAL    = "Global"

    ALL: list[str] = [LOCAL, COMMUNITY, GLOBAL]
    DEFAULT: str   = COMMUNITY

    METHODS: dict[str, list[str]] = {
        LOCAL:     [RetrievalMethod.NODES, RetrievalMethod.TRIPLETS],
        COMMUNITY: [RetrievalMethod.NODES, RetrievalMethod.TRIPLETS,
                    RetrievalMethod.PATHS],
        GLOBAL:    [RetrievalMethod.PATHS, RetrievalMethod.SUBGRAPH],
    }

    STRATEGY: dict[str, str] = {
        LOCAL:     GenerationStrategy.PRE,
        COMMUNITY: GenerationStrategy.MID,
        GLOBAL:    GenerationStrategy.POST,
    }


class Limit:
    FALLBACK_NODES        = 10
    FALLBACK_TRIPLETS     = 50
    PATH_HOPS_MAX         = 3
    SUBGRAPH_HOPS_MAX     = 2

    NODE_QUERY     = 50
    TRIPLET_QUERY  = 50
    PATH_QUERY     = 30
    SUBGRAPH_QUERY = 10

    FALLBACK_NAMES_COUNT  = 3

    SCENE_SUMMARY_MAX_LEN = 200
