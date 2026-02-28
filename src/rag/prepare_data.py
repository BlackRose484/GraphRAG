"""
Prepare vector-store data from the Neo4j Chèo knowledge graph.

Extracts four chunk types:
 - characters  →  "Nhân vật <name>: <description>"
 - actors      →  "Diễn viên <name>"
 - plays       →  "Vở chèo <title>"
 - scenes      →  "Trích đoạn <name>: <summary>"

Then embeds them and saves a ``SimpleVectorStore`` pickle to
``data/vector_store.pkl`` (relative to the project root).

CLI usage
---------
    python -m src.rag.prepare_data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.constants.constant import NodeProp, NodeType, RelType
from src.graph_loader.neo4j_client import Neo4jClient
from src.rag.vector_store import SimpleVectorStore

logger = logging.getLogger(__name__)

# Default output path (project-root/data/vector_store.pkl)
_DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "vector_store.pkl"


# ── Chunk extractors ──────────────────────────────────────────────────────────

def _character_chunks(client: Neo4jClient) -> List[Dict]:
    cypher = (
        f"MATCH (c:{NodeType.CHARACTER}) "
        f"RETURN c.{NodeProp.CHAR_NAME} AS name, c.description AS description "
        f"ORDER BY c.{NodeProp.CHAR_NAME}"
    )
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        desc = r.get("description") or "Không có mô tả"
        chunks.append(
            {
                "text": f"Nhân vật {name}: {desc}",
                "metadata": {"type": NodeType.CHARACTER.lower(), "name": name},
            }
        )
    logger.info("Characters: %d chunks", len(chunks))
    return chunks


def _actor_chunks(client: Neo4jClient) -> List[Dict]:
    cypher = (
        f"MATCH (a:{NodeType.ACTOR}) "
        f"RETURN a.{NodeProp.ACTOR_NAME} AS name "
        f"ORDER BY a.{NodeProp.ACTOR_NAME}"
    )
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        chunks.append(
            {
                "text": f"Diễn viên {name}",
                "metadata": {"type": NodeType.ACTOR.lower(), "name": name},
            }
        )
    logger.info("Actors: %d chunks", len(chunks))
    return chunks


def _play_chunks(client: Neo4jClient) -> List[Dict]:
    cypher = (
        f"MATCH (p:{NodeType.PLAY}) "
        f"RETURN p.{NodeProp.TITLE} AS name "
        f"ORDER BY p.{NodeProp.TITLE}"
    )
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        chunks.append(
            {
                "text": f"Vở chèo {name}",
                "metadata": {"type": NodeType.PLAY.lower(), "name": name},
            }
        )
    logger.info("Plays: %d chunks", len(chunks))
    return chunks


def _scene_chunks(client: Neo4jClient) -> List[Dict]:
    cypher = (
        f"MATCH (s:{NodeType.SCENE}) "
        f"RETURN s.{NodeProp.SCENE_NAME} AS name, "
        f"       s.{NodeProp.SCENE_SUMMARY} AS summary "
        f"ORDER BY s.{NodeProp.SCENE_NAME}"
    )
    chunks = []
    for r in client.read(cypher):
        name    = r.get("name") or ""
        summary = r.get("summary") or "Không có tóm tắt"
        chunks.append(
            {
                "text": f"Trích đoạn {name}: {summary}",
                "metadata": {"type": NodeType.SCENE.lower(), "name": name},
            }
        )
    logger.info("Scenes: %d chunks", len(chunks))
    return chunks


def _relationship_chunks(client: Neo4jClient) -> List[Dict]:
    """
    Character–character relationships (bidirectional via all rel types).
    """
    cypher = (
        f"MATCH (a:{NodeType.CHARACTER})-[r]->(b:{NodeType.CHARACTER}) "
        f"RETURN a.{NodeProp.CHAR_NAME} AS src, type(r) AS rel, "
        f"       b.{NodeProp.CHAR_NAME} AS dst "
        f"LIMIT 200"
    )
    chunks = []
    for r in client.read(cypher):
        src = r.get("src") or ""
        rel = r.get("rel") or ""
        dst = r.get("dst") or ""
        chunks.append(
            {
                "text": f"Quan hệ: {src} —[{rel}]→ {dst}",
                "metadata": {
                    "type": "relationship",
                    "src": src,
                    "dst": dst,
                    "rel": rel,
                },
            }
        )
    logger.info("Relationships: %d chunks", len(chunks))
    return chunks


# ── Public builder ────────────────────────────────────────────────────────────

def build_vector_store(
    output_path: str | Path = _DEFAULT_OUT,
    client: Neo4jClient | None = None,
) -> SimpleVectorStore:
    """
    Pull all chunks from Neo4j, embed them, save, and return the store.

    Args:
        output_path: Where to pickle the finished store.
        client:      Existing ``Neo4jClient``; a new one is created if *None*.

    Returns:
        A fully populated ``SimpleVectorStore``.
    """
    output_path = Path(output_path)
    close_client = client is None
    if close_client:
        client = Neo4jClient()

    try:
        logger.info("Extracting chunks from Neo4j …")
        all_chunks: List[Dict] = []
        all_chunks.extend(_character_chunks(client))
        all_chunks.extend(_actor_chunks(client))
        all_chunks.extend(_play_chunks(client))
        all_chunks.extend(_scene_chunks(client))
        all_chunks.extend(_relationship_chunks(client))
        logger.info("Total chunks extracted: %d", len(all_chunks))

        store = SimpleVectorStore()
        store.add_chunks(all_chunks)
        store.save(output_path)
        return store

    finally:
        if close_client:
            client.close()


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s — %(message)s",
    )
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    build_vector_store(output_path=out)
    print(f"Done → {out}")
