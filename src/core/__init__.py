"""
Core layer: settings, base classes, shared abstractions.

Public API::

    from src.core import settings
    from src.core import BaseModel, BaseRetriever, BaseGenerator, BaseLoader
"""

from src.core.settings import (
    Settings,
    LLMSettings,
    Neo4jSettings,
    OntologySettings,
    settings,
)
from src.core.base import (
    BaseModel,
    BaseRetriever,
    BaseGenerator,
    BaseLoader,
)

__all__ = [
    # Settings
    "Settings",
    "LLMSettings",
    "Neo4jSettings",
    "OntologySettings",
    "settings",
    # Base classes
    "BaseModel",
    "BaseRetriever",
    "BaseGenerator",
    "BaseLoader",
]
