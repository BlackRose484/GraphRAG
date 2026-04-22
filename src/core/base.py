"""
Base classes and abstract interfaces for all GraphRAG components.

Uses LiteLLM as the provider-agnostic LLM layer — no provider SDK is imported
directly here. Switching models only requires changing settings.llm.model in .env.

Supported providers (via LiteLLM prefix):
    gemini/...      → Google Gemini
    anthropic/...   → Anthropic Claude
    openai/...      → OpenAI GPT
    ollama/...      → local Ollama
    openrouter/...  → OpenRouter
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

# Suppress litellm's verbose internal logging
litellm.suppress_debug_info = True

# Auto-drop params the target model doesn't support (e.g. GPT-5 rejects
# temperature=0.0 — only temperature=1 is allowed). Without this flag, RAGAs
# metrics that pass temperature=0 for reproducibility would crash on gpt-5.
litellm.drop_params = True


# ── LLM Base ─────────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """
    Base class for any component that calls an LLM.

    Backed by LiteLLM — works with Gemini, OpenAI, Anthropic, Ollama, and
    100+ other providers. The provider is selected at runtime via
    ``settings.llm.model`` (e.g. ``'gemini/gemini-2.0-flash'``).

    Provides:
    - ``safe_generate(prompt)``  — text in, text out, with exponential-backoff retry
    - ``safe_embed(texts)``      — list of texts → list of embedding vectors

    Usage::

        class MyQueryProcessor(BaseModel):
            def run(self, query: str) -> str:
                return self.safe_generate(f"Expand: {query}")
    """

    # Exception types LiteLLM raises for transient failures
    _RETRYABLE = (RateLimitError, ServiceUnavailableError, Timeout, APIConnectionError)

    def __init__(self, model_name: Optional[str] = None) -> None:
        # When None, ``self.model_name`` resolves dynamically from
        # ``settings.llm.model`` on every access — this lets cached pipeline
        # instances pick up runtime model switches (e.g. the benchmark UI's
        # ``override_model`` context manager).
        self._model_name: Optional[str] = model_name

    @property
    def model_name(self) -> str:
        return self._model_name or settings.llm.model

    # ── Text generation ───────────────────────────────────────────────────────

    def safe_generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Call the LLM with automatic retry on transient failures.

        Converts the prompt string to a single user message (OpenAI chat format,
        which LiteLLM translates to the correct format for every provider).

        Args:
            prompt: The full prompt string.
            temperature: Per-call temperature override. When ``None`` (default),
                ``settings.llm.temperature`` is used. Pass ``0.0`` for
                deterministic output (e.g. LLM-judge metrics).

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
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
                # Non-retryable — fail immediately
                raise RuntimeError(
                    f"LLM generation failed (non-retryable): {exc}"
                ) from exc

        raise RuntimeError(
            f"LLM generation failed after {settings.llm.max_retries} attempt(s): "
            f"{last_error}"
        )

    # ── Embeddings ────────────────────────────────────────────────────────────

    # Gemini's batchEmbedContents caps at 100 per request; other providers allow
    # more but 100 is a universally safe chunk size.
    _EMBED_BATCH_SIZE = 100

    def safe_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the configured embedding model.

        Automatically splits the input into batches of ``_EMBED_BATCH_SIZE`` to
        respect provider limits (e.g. Gemini caps at 100 per request).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (one per input text, in the same order).

        Raises:
            RuntimeError: On API error.
        """
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


# ── Abstract Interfaces ───────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Abstract interface every G-Retrieval component must implement.

    The ``retrieve`` call is intentionally generic — each implementation
    specifies concrete kwarg names in its own docstring.
    """

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve relevant graph data for *query*.

        Returns:
            A dict whose keys are retrieval-method names (e.g. ``"nodes"``,
            ``"triplets"``) mapping to lists of result objects.
        """


class BaseGenerator(ABC):
    """
    Abstract interface every G-Generation component must implement.
    """

    @abstractmethod
    def generate(self, query: str, context: Dict[str, Any], **kwargs) -> str:
        """
        Generate a natural-language answer for *query* given *context*.

        Args:
            query:   The original user question.
            context: Formatted graph data produced by ``GraphFormatConverter``.

        Returns:
            The final answer string.
        """


class BaseLoader(ABC):
    """
    Abstract interface for data loaders (e.g. Neo4j ontology loader).

    Separating loading from retrieval keeps both layers independently testable.
    """

    @abstractmethod
    def load(self, **kwargs) -> bool:
        """
        Load data into the target store.

        Returns:
            True on success, False on partial failure (with logged details).
        """


# ── Shared Data Transfer Object ───────────────────────────────────────────────

@dataclass
class ProcessingResult:
    """
    Generic result envelope passed between pipeline stages.

    Attributes:
        success:  Whether the operation completed without error.
        data:     The primary result payload (type varies per stage).
        error:    Human-readable error message when ``success`` is False.
        metadata: Arbitrary key-value pairs (timings, counts, etc.).
    """

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (useful for JSON logging)."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }
