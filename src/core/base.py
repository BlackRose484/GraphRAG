"""Base classes and abstract interfaces for all GraphRAG components.

Uses LiteLLM as the provider-agnostic LLM layer — switching models only
requires changing settings.llm.model in .env.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import litellm
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    APIConnectionError,
)

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

litellm.suppress_debug_info = True

# Auto-drop params the target model doesn't support (e.g. GPT-5 only allows
# temperature=1). Without this flag, RAGAs metrics that pass temperature=0
# would crash on gpt-5.
litellm.drop_params = True


class BaseModel(ABC):
    """Base class for any component that calls an LLM via LiteLLM."""

    _RETRYABLE = (RateLimitError, ServiceUnavailableError, Timeout, APIConnectionError)

    def __init__(self, model_name: Optional[str] = None) -> None:
        # When None, ``self.model_name`` resolves dynamically from
        # ``settings.llm.model`` so cached instances pick up runtime model
        # switches (e.g. benchmark UI's ``override_model`` context manager).
        self._model_name: Optional[str] = model_name

    @property
    def model_name(self) -> str:
        return self._model_name or settings.llm.model

    def safe_generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call the LLM with exponential-backoff retry on transient failures.

        ``temperature=None`` falls back to ``settings.llm.temperature``. Pass
        ``0.0`` for deterministic output (e.g. LLM-judge metrics).
        """
        messages = [{"role": "user", "content": prompt}]
        last_error: Optional[Exception] = None
        temp = temperature if temperature is not None else settings.llm.temperature

        for attempt in range(settings.llm.max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    timeout=settings.llm.timeout,
                )
                return response.choices[0].message.content or ""

            except self._RETRYABLE as exc:
                last_error = exc
                if attempt == settings.llm.max_retries - 1:
                    break
                delay = settings.llm.retry_delay * (2 ** attempt)
                logger.warning(
                    "Retryable LLM error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    settings.llm.max_retries,
                    str(exc)[:120],
                    delay,
                )
                time.sleep(delay)

            except Exception as exc:
                raise RuntimeError(
                    f"LLM generation failed (non-retryable): {exc}"
                ) from exc

        raise RuntimeError(
            f"LLM generation failed after {settings.llm.max_retries} attempt(s): "
            f"{last_error}"
        )

    # Gemini's batchEmbedContents caps at 100 per request; other providers allow
    # more but 100 is a universally safe chunk size.
    _EMBED_BATCH_SIZE = 100

    def safe_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, batching to respect provider limits."""
        if not texts:
            return []

        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._EMBED_BATCH_SIZE):
            batch = texts[i : i + self._EMBED_BATCH_SIZE]
            try:
                response = litellm.embedding(
                    model=settings.llm.embedding_model,
                    input=batch,
                )
            except Exception as exc:
                raise RuntimeError(f"Embedding failed: {exc}") from exc
            vectors.extend(item["embedding"] for item in response["data"])

        return vectors


class BaseRetriever(ABC):
    """Abstract interface every G-Retrieval component must implement."""

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """Retrieve relevant graph data for *query*."""


class BaseGenerator(ABC):
    """Abstract interface every G-Generation component must implement."""

    @abstractmethod
    def generate(self, query: str, context: Dict[str, Any], **kwargs) -> str:
        """Generate a natural-language answer for *query* given *context*."""


class BaseLoader(ABC):
    """Abstract interface for data loaders (e.g. Neo4j ontology loader)."""

    @abstractmethod
    def load(self, **kwargs) -> bool:
        """Load data into the target store."""
