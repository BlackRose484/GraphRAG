"""
SimpleVectorStore — in-memory vector store backed by LiteLLM embeddings.

Embedding is delegated to ``BaseModel.safe_embed`` so the embedding model
is configured centrally via ``LLM_EMBEDDING_MODEL`` (default:
``gemini/text-embedding-004``).

Persistent storage uses pickle so the store can be built once and reused
across runs without re-embedding.
"""

from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.core.base import BaseModel

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """
    In-memory vector store with cosine-similarity search.

    Usage
    -----
    >>> store = SimpleVectorStore()
    >>> store.add_chunks([{"text": "...", "metadata": {"type": "character"}}])
    >>> results = store.query("Trần Phương là ai?", top_k=5)
    """

    def __init__(self) -> None:
        self._base: BaseModel = BaseModel()
        self._chunks:     List[str]             = []
        self._embeddings: List[np.ndarray]      = []
        self._metadata:   List[Dict[str, Any]]  = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embed and store a list of text chunks.

        Args:
            chunks: Each item must have a ``"text"`` key and an optional
                    ``"metadata"`` dict.
        """
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks …", len(texts))

        vectors = self._base.safe_embed(texts)

        for chunk, vec in zip(chunks, vectors):
            self._chunks.append(chunk["text"])
            self._embeddings.append(np.array(vec, dtype=np.float32))
            self._metadata.append(chunk.get("metadata", {}))

        logger.info("Store now contains %d chunks.", len(self._chunks))

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed *question* and return the *top_k* most-similar chunks.

        Returns:
            List of dicts with ``"text"``, ``"metadata"``, and ``"score"``
            keys, sorted by descending similarity.
        """
        if not self._chunks:
            return []

        vecs = self._base.safe_embed([question])
        q_vec = np.array(vecs[0], dtype=np.float32)

        sims = [self._cosine(q_vec, e) for e in self._embeddings]
        top_idxs = np.argsort(sims)[-top_k:][::-1]

        return [
            {
                "text":     self._chunks[i],
                "metadata": self._metadata[i],
                "score":    float(sims[i]),
            }
            for i in top_idxs
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str | Path) -> None:
        """Pickle the store to *filepath*."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as fh:
            pickle.dump(
                {
                    "chunks":     self._chunks,
                    "embeddings": self._embeddings,
                    "metadata":   self._metadata,
                },
                fh,
            )
        logger.info("VectorStore saved → %s  (%d chunks)", filepath, len(self._chunks))

    @classmethod
    def load(cls, filepath: str | Path) -> "SimpleVectorStore":
        """Restore a pickled store from *filepath*."""
        filepath = Path(filepath)
        store = cls()
        with filepath.open("rb") as fh:
            data = pickle.load(fh)
        store._chunks     = data["chunks"]
        store._embeddings = data["embeddings"]
        store._metadata   = data["metadata"]
        logger.info("VectorStore loaded ← %s  (%d chunks)", filepath, len(store._chunks))
        return store

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def __len__(self) -> int:
        return len(self._chunks)
