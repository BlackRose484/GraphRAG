"""
Package: src.constants

Re-exports every domain constant and prompt template so callers can simply do:
    from src.constants import NodeType, RelType, FormatKey, PRE_GENERATION, ...
"""

from .constant import (
    ONTOLOGY_NAMESPACE,
    NodeType,
    RelType,
    NodeProp,
    EntityType,
    RetrievalMethod,
    FormatKey,
    GenerationStrategy,
    Limit,
    CHARACTER_SUBCLASS_TO_ROLE,
)

from .prompt_engineer import (
    QUERY_EXPAND,
    QUERY_DECOMPOSE,
    ENTITY_EXTRACT,
    PRE_GENERATION,
    MID_GENERATION,
    POST_INITIAL,
    POST_REFINE,
    CONTEXT_HEADER,
    CONTEXT_KEY_FACTS_HEADER,
    CONTEXT_SECTION,
)

__all__ = [
    # constant.py
    "ONTOLOGY_NAMESPACE",
    "NodeType",
    "RelType",
    "NodeProp",
    "EntityType",
    "RetrievalMethod",
    "FormatKey",
    "GenerationStrategy",
    "Limit",
    "CHARACTER_SUBCLASS_TO_ROLE",
    # prompt_engineer.py
    "QUERY_EXPAND",
    "QUERY_DECOMPOSE",
    "ENTITY_EXTRACT",
    "PRE_GENERATION",
    "MID_GENERATION",
    "POST_INITIAL",
    "POST_REFINE",
    "CONTEXT_HEADER",
    "CONTEXT_KEY_FACTS_HEADER",
    "CONTEXT_SECTION",
]
