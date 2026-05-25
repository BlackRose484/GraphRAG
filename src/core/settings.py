"""Settings management — singleton, type-safe, .env-backed.

Model names follow the LiteLLM provider-prefix format (e.g.
``gemini/gemini-2.0-flash``, ``anthropic/claude-3-5-sonnet-20241022``).
Switching providers only requires updating LLM_MODEL + the matching API key.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from dotenv import load_dotenv

# Curated defaults — used by LLMSettings.available_models when LLM_MODELS_AVAILABLE
# is not set. Kept conservative so the dropdown doesn't offer models LiteLLM 404s on.
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


@dataclass
class LLMSettings:
    """Provider-agnostic LLM configuration, routed via LiteLLM."""

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
        return self.model.split("/")[0] if "/" in self.model else self.model

    def __post_init__(self) -> None:
        ok, msg = self.validate_api_key()
        if not ok:
            _logger.warning(msg)

    def validate_api_key(self) -> tuple[bool, str]:
        """Check if the required API key env var is set."""
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
            return True
        return bool(os.getenv(required))

    @property
    def available_models(self) -> List[str]:
        """Models the user can pick at runtime.

        Source: LLM_MODELS_AVAILABLE allowlist if set, else providers with a
        valid API key from _DEFAULT_MODEL_CATALOG. The current model is always
        included so it stays selectable.
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
        seen: set = set()
        ordered: List[str] = []
        for m in [self.model, *listed]:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        return [m for m in ordered if self.has_key_for(m)]

    @property
    def available_models_source(self) -> str:
        """``"allowlist"`` when LLM_MODELS_AVAILABLE is set, else ``"auto"``."""
        return "allowlist" if os.getenv("LLM_MODELS_AVAILABLE", "").strip() else "auto"

    @contextmanager
    def override_model(self, model_name: Optional[str]) -> Iterator[str]:
        """Temporarily swap ``settings.llm.model`` for the duration of a block.

        BaseModel instances see the override because ``model_name`` reads
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
        return bool(self.password)


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
        return bool(self.sender and self.app_password and self.receiver)


@dataclass
class AppSettings:
    """Application-level toggles for guest-mode and admin unlock."""

    guest_mode: bool = field(
        default_factory=lambda: os.getenv("GUEST_MODE", "").strip().lower()
        in ("1", "true", "yes", "on"),
    )
    admin_password: str = field(
        default_factory=lambda: os.getenv("ADMIN_PASSWORD", "").strip(),
    )

    @property
    def admin_unlock_available(self) -> bool:
        return bool(self.admin_password)


@dataclass
class OntologySettings:
    """Chèo domain ontology file configuration."""

    file_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "CheoOntology_v4.ttl"
    )
    namespace: str = "http://www.semanticweb.org/asus/ontologies/2025/5/Cheo#"

    def __post_init__(self) -> None:
        pass  # File existence is checked lazily when the ontology is actually loaded


class Settings:
    """Application-wide settings — singleton."""

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
        self.gmail = GmailSettings()
        self.ontology = OntologySettings()
        self.app = AppSettings()
        self._initialised = True

    @classmethod
    def get_instance(cls) -> Settings:
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


settings = Settings.get_instance()
