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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from dotenv import load_dotenv

# Curated default models per provider — used by LLMSettings.available_models
# when the user has not supplied an explicit LLM_MODELS_AVAILABLE allowlist.
# Kept conservative (well-known stable names) so the dropdown doesn't offer
# models LiteLLM will 404 on.
_DEFAULT_MODEL_CATALOG: dict[str, list[str]] = {
    "openai": [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-5-mini",
    ],
    "anthropic": [
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-5-sonnet-20241022",
    ],
    "gemini": [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
    ],
    "ollama": [
        "ollama/llama3.2",
    ],
}

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

    def _provider_of(self, model_name: str) -> str:
        return model_name.split("/")[0] if "/" in model_name else model_name

    def has_key_for(self, model_name: str) -> bool:
        """True if the provider of ``model_name`` has its API key configured."""
        required = self._PROVIDER_KEY_MAP.get(self._provider_of(model_name))
        if required is None:
            return True  # e.g. ollama — no key needed
        return bool(os.getenv(required))

    @property
    def available_models(self) -> List[str]:
        """Models the user can pick at runtime.

        Two sources, in priority order:

        1. **User allowlist** — ``LLM_MODELS_AVAILABLE`` in .env (comma-
           separated). When set, only these are offered (plus the current
           model so it's always selectable).

        2. **Auto-detect** — when the allowlist is empty, every model in
           :data:`_DEFAULT_MODEL_CATALOG` whose provider has a valid API
           key is offered. This means configuring ``GEMINI_API_KEY`` and
           ``OPENAI_API_KEY`` in .env is enough to see both providers'
           models in the dropdown without any extra config.

        Final list is always filtered to providers with a valid API key.
        """
        raw = os.getenv("LLM_MODELS_AVAILABLE", "").strip()
        if raw:
            listed = [m.strip() for m in raw.split(",") if m.strip()]
        else:
            listed = [
                m for provider, models in _DEFAULT_MODEL_CATALOG.items()
                for m in models
                if self.has_key_for(f"{provider}/_probe")
            ]
        # Preserve order, ensure current model is first and deduped
        seen: set = set()
        ordered: List[str] = []
        for m in [self.model, *listed]:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        return [m for m in ordered if self.has_key_for(m)]

    @property
    def available_models_source(self) -> str:
        """Which branch of ``available_models`` produced the current list.

        Returns ``"allowlist"`` when ``LLM_MODELS_AVAILABLE`` is set,
        ``"auto"`` otherwise. Useful for UI hints.
        """
        return "allowlist" if os.getenv("LLM_MODELS_AVAILABLE", "").strip() else "auto"

    @contextmanager
    def override_model(self, model_name: Optional[str]) -> Iterator[str]:
        """Temporarily swap ``settings.llm.model`` for the duration of a block.

        Usage::

            with settings.llm.override_model("anthropic/claude-3-5-haiku-20241022"):
                runner.run(...)   # all pipelines + metrics now use Claude

        Pass ``None`` or the current model to no-op. Restores on exit even if
        the block raises. BaseModel instances created inside (or already cached
        outside) will see the override because ``BaseModel.model_name`` reads
        ``settings.llm.model`` dynamically when not pinned at init.
        """
        if not model_name or model_name == self.model:
            yield self.model
            return
        original = self.model
        self.model = model_name
        _logger.info("LLM model override: %s → %s", original, model_name)
        try:
            yield model_name
        finally:
            self.model = original
            _logger.info("LLM model restored: %s", original)


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
class GmailSettings:
    """Gmail SMTP configuration for sending experiment results."""

    sender: str = field(
        default_factory=lambda: os.getenv("EMAIL_USER", "")
    )
    app_password: str = field(
        default_factory=lambda: os.getenv("EMAIL_PASS", "")
    )
    receiver: str = field(
        default_factory=lambda: os.getenv("ADMIN_EMAIL", "")
    )

    @property
    def is_configured(self) -> bool:
        """True nếu đủ 3 biến môi trường EMAIL_USER / EMAIL_PASS / ADMIN_EMAIL."""
        return bool(self.sender and self.app_password and self.receiver)


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
        self.gmail = GmailSettings()
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
