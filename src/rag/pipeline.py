"""
VectorRAGPipeline — traditional retrieval-augmented generation using a
pre-built ``SimpleVectorStore``.

Architecture
------------
 query
   │
   ├─ 1. embed query              (BaseModel.safe_embed)
   ├─ 2. vector search top-k      (SimpleVectorStore.query)
   ├─ 3. build graph_data struct  (metrics-compatible)
   ├─ 4. format context string
   └─ 5. LLM generation           (BaseModel.safe_generate)

The ``PipelineResult`` dataclass mirrors the structure used by the
GraphRAG pipeline so the benchmark module can evaluate both systems
with the same metrics code.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.base import BaseModel
from src.core.settings import settings
from src.rag.vector_store import SimpleVectorStore

logger = logging.getLogger(__name__)

# Default store path lives in project-root/data/
_DEFAULT_STORE = Path(__file__).resolve().parents[3] / "data" / "vector_store.pkl"
_TOP_K_DEFAULT = 5


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Retrieval step output (metrics-compatible with GraphRAG result)."""
    query:             str
    graph_data:        Dict[str, Any]
    retrieval_methods: List[str]
    format_converters: List[str]
    num_nodes:         int
    num_triplets:      int
    num_paths:         int
    retrieval_time:    float
    formatted_contexts: Dict[str, str]
    key_facts:         str
    error:             Optional[str] = None


@dataclass
class GenerationResult:
    """Generation step output."""
    query:            str
    response:         str
    strategy:         str
    generation_time:  float
    context_length:   int
    key_facts_used:   str
    model_name:       str
    prompt_template:  str
    error:            Optional[str] = None


@dataclass
class PipelineResult:
    """Complete RAG pipeline result."""
    retrieval:  RetrievalResult
    generation: GenerationResult

    @property
    def answer(self) -> str:
        """Shortcut — the generated answer text."""
        return self.generation.response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval":  asdict(self.retrieval),
            "generation": asdict(self.generation),
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class VectorRAGPipeline(BaseModel):
    """
    End-to-end traditional RAG pipeline.

    Args:
        vector_store: Pre-built ``SimpleVectorStore``.  If *None* the pipeline
                      tries to load from *store_path*.
        store_path:   Path to a pickled store; ignored when *vector_store* is
                      already supplied.
        top_k:        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        vector_store: Optional[SimpleVectorStore] = None,
        store_path:   str | Path = _DEFAULT_STORE,
        top_k:        int = _TOP_K_DEFAULT,
    ) -> None:
        super().__init__()
        self._top_k = top_k

        if vector_store is not None:
            self._store = vector_store
        else:
            path = Path(store_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {path}. "
                    "Run `python -m src.rag.prepare_data` first."
                )
            self._store = SimpleVectorStore.load(path)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, question: str, top_k: Optional[int] = None) -> PipelineResult:
        """
        Execute the full RAG pipeline for *question*.

        Args:
            question: User query (Vietnamese natural language).
            top_k:    Override the instance-level ``top_k`` for this call.

        Returns:
            A ``PipelineResult`` with full retrieval and generation details.
        """
        k = top_k if top_k is not None else self._top_k

        # ── Retrieval ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            chunks = self._store.query(question, top_k=k)
            retrieval_error: Optional[str] = None
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            chunks = []
            retrieval_error = str(exc)
        retrieval_time = time.time() - t0

        graph_data = self._build_graph_data(chunks)
        context    = self._format_context(chunks)
        key_facts  = self._extract_key_facts(chunks)

        retrieval = RetrievalResult(
            query              = question,
            graph_data         = graph_data,
            retrieval_methods  = ["vector_search"],
            format_converters  = ["natural_language"],
            num_nodes          = len(chunks),
            num_triplets       = 0,
            num_paths          = 0,
            retrieval_time     = retrieval_time,
            formatted_contexts = {"natural_language": context},
            key_facts          = key_facts,
            error              = retrieval_error,
        )

        # ── Generation ────────────────────────────────────────────────────────
        t1 = time.time()
        try:
            prompt   = self._build_prompt(question, context)
            response = self.safe_generate(prompt)
            gen_error: Optional[str] = None
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            response  = "Xin lỗi, không thể tạo câu trả lời."
            gen_error = str(exc)
        generation_time = time.time() - t1

        generation = GenerationResult(
            query           = question,
            response        = response,
            strategy        = "vector_rag",
            generation_time = generation_time,
            context_length  = len(context),
            key_facts_used  = key_facts,
            model_name      = self.model_name,
            prompt_template = "rag_template",
            error           = gen_error,
        )

        return PipelineResult(retrieval=retrieval, generation=generation)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_graph_data(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Wrap retrieved chunks in the same ``graph_data`` schema used by the
        GraphRAG pipeline so benchmark metrics work without modification.
        """
        nodes = [
            {
                "id":          f"chunk_{i}",
                "type":        c["metadata"].get("type", "text"),
                "name":        c["metadata"].get("name", f"Chunk {i}"),
                "text":        c["text"],
                "summary":     c["text"][:200],
                "description": c["text"],
                "score":       c.get("score", 0.0),
            }
            for i, c in enumerate(chunks)
        ]
        return {
            "nodes":    nodes,
            "triplets": [],
            "paths":    [],
            "subgraph": {"nodes": nodes, "relationships": []},
        }

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "Không tìm thấy thông tin liên quan."
        parts = [f"[Thông tin {i}]\n{c['text']}" for i, c in enumerate(chunks, 1)]
        return "\n\n".join(parts)

    @staticmethod
    def _extract_key_facts(chunks: List[Dict[str, Any]]) -> str:
        facts = []
        for c in chunks[:3]:
            text = c["text"]
            fact = text.split(".")[0] if "." in text else text[:100]
            facts.append(f"- {fact}")
        return "\n".join(facts)

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "Dựa trên thông tin sau đây, hãy trả lời câu hỏi một cách chính xác và đầy đủ.\n\n"
            f"Thông tin tham khảo:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Hướng dẫn:\n"
            "- Chỉ sử dụng thông tin từ phần \"Thông tin tham khảo\" ở trên\n"
            "- Nếu thông tin không đủ để trả lời, hãy nói rõ\n"
            "- Trả lời ngắn gọn, súc tích nhưng đầy đủ\n\n"
            "Trả lời:"
        )
