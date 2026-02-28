"""
Settings management — singleton, type-safe, .env-backed.

Model names follow the LiteLLM provider-prefix format::

    gemini/gemini-2.0-flash              → Google Gemini
    anthropic/claude-3-5-sonnet-20241022 → Anthropic Claude
    openai/gpt-4o                        → OpenAI
    ollama/llama3.2                      → local Ollama
    openrouter/google/gemini-2.0-flash   → OpenRouter

Switching providers only requires updating LLM_MODEL (and the matching API
key) in .env — no code changes needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Import lazily to avoid circular import — utils depends on nothing in core
from src.utils.logger import get_logger  # noqa: E402
_logger = get_logger(__name__)


# ── LLM Settings ──────────────────────────────────────────────────────────────

@dataclass
class LLMSettings:
    """
    Provider-agnostic LLM configuration, routed via LiteLLM.

    ``model`` and ``embedding_model`` must use LiteLLM's
    ``<provider>/<model-name>`` format.

    API keys are loaded from the standard env vars each provider expects
    (LiteLLM reads them automatically):

        GEMINI_API_KEY      → gemini/...
        ANTHROPIC_API_KEY   → anthropic/...
        OPENAI_API_KEY      → openai/...
        OPENROUTER_API_KEY  → openrouter/...
        (none needed)       → ollama/...
    """

    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_EMBEDDING_MODEL", "gemini/text-embedding-004"
        )
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "300"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3"))
    )
    retry_delay: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_DELAY", "2.0"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )

    # provider prefix → required env var  (None = no key needed)
    _PROVIDER_KEY_MAP: dict = field(
        default_factory=lambda: {
            "gemini":     "GEMINI_API_KEY",
            "anthropic":  "ANTHROPIC_API_KEY",
            "openai":     "OPENAI_API_KEY",
            "azure":      "AZURE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "ollama":     None,
            "vertex_ai":  "GOOGLE_APPLICATION_CREDENTIALS",
        },
        repr=False,
    )

    @property
    def provider(self) -> str:
        """Extract provider prefix, e.g. 'gemini' from 'gemini/gemini-2.0-flash'."""
        return self.model.split("/")[0] if "/" in self.model else self.model

    def __post_init__(self) -> None:
        ok, msg = self.validate_api_key()
        if not ok:
            _logger.warning(msg)

    def validate_api_key(self) -> tuple[bool, str]:
        """Check if the required API key env var is set.

        Returns:
            (True, "") if OK, or (False, error_message) if missing.
        """
        required_env = self._PROVIDER_KEY_MAP.get(self.provider)
        if required_env is not None and not os.getenv(required_env):
            msg = (
                f"Provider '{self.provider}' requires '{required_env}' "
                f"to be set in your .env file."
            )
            return False, msg
        return True, ""


@dataclass
class Neo4jSettings:
    """Neo4j graph database connection configuration."""

    uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j")
    )
    password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )

    def __post_init__(self) -> None:
        pass  # Validation is deferred — checked lazily when Neo4j is actually used

    @property
    def is_configured(self) -> bool:
        """True if a password has been supplied."""
        return bool(self.password)


@dataclass
class ChromaSettings:
    """ChromaDB vector store configuration (optional — not used in main pipeline)."""

    persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db")
        )
    )

    def __post_init__(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class OntologySettings:
    """Chèo domain ontology file configuration."""

    file_path: Path = field(
        # GraphRAGv2/data/CheoOntology.ttl
        default_factory=lambda: _PROJECT_ROOT / "data" / "CheoOntology.ttl"
    )
    namespace: str = "http://www.semanticweb.org/asus/ontologies/2025/5/Cheo#"

    def __post_init__(self) -> None:
        pass  # File existence is checked lazily when the ontology is actually loaded


# ── Main Settings singleton ───────────────────────────────────────────────────

class Settings:
    """
    Application-wide settings — singleton.

    Usage::

        from src.core.settings import settings

        print(settings.llm.model)     # 'gemini/gemini-2.0-flash'
        print(settings.llm.provider)  # 'gemini'
        print(settings.neo4j.uri)
    """

    _instance: Optional[Settings] = None

    def __new__(cls) -> Settings:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialised = False
            cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self.llm = LLMSettings()
        self.neo4j = Neo4jSettings()
        self.chroma = ChromaSettings()
        self.ontology = OntologySettings()
        self._initialised = True

    @classmethod
    def get_instance(cls) -> Settings:
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls()
        return cls._instance  # type: ignore[return-value]

    def validate(self) -> bool:
        """Sanity-check all settings, emit results via logger."""
        ok = True

        api_ok, api_msg = self.llm.validate_api_key()
        if not api_ok:
            _logger.error("LLM config: %s", api_msg)
            ok = False

        if not self.neo4j.password:
            _logger.error("NEO4J_PASSWORD is not set")
            ok = False

        if not self.ontology.file_path.exists():
            _logger.error("Ontology file not found: %s", self.ontology.file_path)
            ok = False

        if ok:
            _logger.info(
                "Settings OK — LLM: %s | Neo4j: %s",
                self.llm.model,
                self.neo4j.uri,
            )
        return ok

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"llm={self.llm.model!r}, "
            f"neo4j={self.neo4j.uri!r}, "
            f"ontology={self.ontology.file_path})"
        )


# Module-level singleton — import this everywhere
settings = Settings.get_instance()
