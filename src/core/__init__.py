"""
Core layer: settings, base classes, shared abstractions.

Public API::

    from src.core import settings
    from src.core import BaseModel, BaseRetriever, BaseGenerator, BaseLoader
    from src.core import ProcessingResult
"""

from src.core.settings import (
    Settings,
    LLMSettings,
    Neo4jSettings,
    ChromaSettings,
    OntologySettings,
    settings,
)
from src.core.base import (
    BaseModel,
    BaseRetriever,
    BaseGenerator,
    BaseLoader,
    ProcessingResult,
)

__all__ = [
    # Settings
    "Settings",
    "LLMSettings",
    "Neo4jSettings",
    "ChromaSettings",
    "OntologySettings",
    "settings",
    # Base classes
    "BaseModel",
    "BaseRetriever",
    "BaseGenerator",
    "BaseLoader",
    "ProcessingResult",
]
