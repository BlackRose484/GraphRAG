"""
Traditional Vector-RAG module for GraphRAGv2.

Public API
----------
SimpleVectorStore   — in-memory store with LiteLLM embeddings + cosine search
build_vector_store  — pull chunks from Neo4j and embed them
VectorRAGPipeline   — end-to-end RAG: embed query → search → generate
"""

from .vector_store import SimpleVectorStore
from .prepare_data import build_vector_store
from .pipeline import VectorRAGPipeline, PipelineResult

__all__ = [
    "SimpleVectorStore",
    "build_vector_store",
    "VectorRAGPipeline",
    "PipelineResult",
]
