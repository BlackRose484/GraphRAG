"""
Prepare vector-store data from the Neo4j Chèo knowledge graph.

Produces one chunk per main entity (Character / Actor / Play / Scene) where
each chunk text bundles the entity's relationships so vector search over
natural-language queries can answer relational questions like
"Ai đóng vai Thị Mầu?" or "Các nhân vật trong vở Kim Nham là ai?".

Then embeds them and saves a ``SimpleVectorStore`` pickle to
``data/vector_store.pkl`` (relative to the project root).

CLI usage
---------
    python -m src.rag.prepare_data
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.constants.constant import NodeProp, NodeType
from src.graph_loader.neo4j_client import Neo4jClient
from src.rag.vector_store import SimpleVectorStore

logger = logging.getLogger(__name__)

# Default output path (project-root/data/vector_store.pkl)
_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "vector_store.pkl"


# ── Chunk extractors ──────────────────────────────────────────────────────────

def _character_chunks(client: Neo4jClient) -> List[Dict]:
    """
    Character chunk = description + plays it belongs to + actors who played it.
    """
    cypher = f"""
    MATCH (c:{NodeType.CHARACTER})
    OPTIONAL MATCH (p:{NodeType.PLAY})-[:HAS_CHARACTER]->(c)
    OPTIONAL MATCH (c)<-[:FOR_CHARACTER]-(:{NodeType.ROLE_ASSIGNMENT})-[:PERFORMED_BY]->(a:{NodeType.ACTOR})
    RETURN c.{NodeProp.CHAR_NAME}   AS name,
           c.description             AS description,
           collect(DISTINCT p.{NodeProp.TITLE}) AS plays,
           collect(DISTINCT a.{NodeProp.ACTOR_NAME}) AS actors
    ORDER BY name
    """
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        desc = r.get("description") or "Không có mô tả"
        plays = [p for p in (r.get("plays") or []) if p]
        actors = [a for a in (r.get("actors") or []) if a]

        parts = [f"Nhân vật {name}: {desc}"]
        if plays:
            parts.append(f"Xuất hiện trong vở: {', '.join(plays)}.")
        if actors:
            parts.append(f"Các diễn viên đã thủ vai {name}: {', '.join(actors)}.")

        chunks.append({
            "text": " ".join(parts),
            "metadata": {"type": NodeType.CHARACTER.lower(), "name": name},
        })
    logger.info("Characters: %d chunks", len(chunks))
    return chunks


def _actor_chunks(client: Neo4jClient) -> List[Dict]:
    """
    Actor chunk = list of (character, play) pairs the actor has performed.
    This is the key chunk for answering "Ai đóng vai X?" /
    "Diễn viên Y đóng những vai gì?".
    """
    cypher = f"""
    MATCH (a:{NodeType.ACTOR})
    OPTIONAL MATCH (a)<-[:PERFORMED_BY]-(ra:{NodeType.ROLE_ASSIGNMENT})-[:FOR_CHARACTER]->(c:{NodeType.CHARACTER})
    OPTIONAL MATCH (p:{NodeType.PLAY})-[:HAS_CHARACTER]->(c)
    RETURN a.{NodeProp.ACTOR_NAME}   AS name,
           collect(DISTINCT {{char: c.{NodeProp.CHAR_NAME},
                              play: p.{NodeProp.TITLE}}}) AS roles
    ORDER BY name
    """
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        roles = [
            (item["char"], item["play"])
            for item in (r.get("roles") or [])
            if item and item.get("char")
        ]

        parts = [f"Diễn viên {name}."]
        if roles:
            # Group characters by play for readability
            by_play: dict[str, list[str]] = defaultdict(list)
            for char, play in roles:
                by_play[play or "(không rõ vở)"].append(char)
            role_descs = [
                f"vở {play}: {', '.join(sorted(set(chars)))}"
                for play, chars in sorted(by_play.items())
            ]
            parts.append(f"Đã thủ các vai — {'; '.join(role_descs)}.")
        else:
            parts.append("Chưa có thông tin vai diễn.")

        chunks.append({
            "text": " ".join(parts),
            "metadata": {"type": NodeType.ACTOR.lower(), "name": name},
        })
    logger.info("Actors: %d chunks", len(chunks))
    return chunks


def _play_chunks(client: Neo4jClient) -> List[Dict]:
    """
    Play chunk = list of characters + list of scenes in the play.
    Answers "Liệt kê nhân vật trong vở X" / "Vở X có những trích đoạn nào?".
    """
    cypher = f"""
    MATCH (p:{NodeType.PLAY})
    OPTIONAL MATCH (p)-[:HAS_CHARACTER]->(c:{NodeType.CHARACTER})
    OPTIONAL MATCH (p)-[:HAS_SCENE]->(s:{NodeType.SCENE})
    RETURN p.{NodeProp.TITLE}       AS name,
           collect(DISTINCT c.{NodeProp.CHAR_NAME})  AS characters,
           collect(DISTINCT s.{NodeProp.SCENE_NAME}) AS scenes
    ORDER BY name
    """
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        characters = sorted([c for c in (r.get("characters") or []) if c])
        scenes = sorted([s for s in (r.get("scenes") or []) if s])

        parts = [f"Vở chèo {name}."]
        if characters:
            parts.append(f"Các nhân vật trong vở: {', '.join(characters)}.")
        if scenes:
            parts.append(f"Các trích đoạn của vở: {', '.join(scenes)}.")

        chunks.append({
            "text": " ".join(parts),
            "metadata": {"type": NodeType.PLAY.lower(), "name": name},
        })
    logger.info("Plays: %d chunks", len(chunks))
    return chunks


def _scene_chunks(client: Neo4jClient) -> List[Dict]:
    """
    Scene chunk = summary + parent play + characters that appear +
    actors that performed in it.
    """
    cypher = f"""
    MATCH (s:{NodeType.SCENE})
    OPTIONAL MATCH (p:{NodeType.PLAY})-[:HAS_SCENE]->(s)
    OPTIONAL MATCH (s)-[:HAS_VERSION]->(v:{NodeType.VERSION})
                   <-[:IN_VERSION]-(ra:{NodeType.ROLE_ASSIGNMENT})
    OPTIONAL MATCH (ra)-[:FOR_CHARACTER]->(c:{NodeType.CHARACTER})
    OPTIONAL MATCH (ra)-[:PERFORMED_BY]->(a:{NodeType.ACTOR})
    RETURN s.{NodeProp.SCENE_NAME}    AS name,
           s.{NodeProp.SCENE_SUMMARY} AS summary,
           head(collect(DISTINCT p.{NodeProp.TITLE}))   AS play,
           collect(DISTINCT c.{NodeProp.CHAR_NAME})     AS characters,
           collect(DISTINCT a.{NodeProp.ACTOR_NAME})    AS actors
    ORDER BY name
    """
    chunks = []
    for r in client.read(cypher):
        name = r.get("name") or ""
        summary = r.get("summary") or "Không có tóm tắt"
        play = r.get("play") or ""
        characters = sorted([c for c in (r.get("characters") or []) if c])
        actors = sorted([a for a in (r.get("actors") or []) if a])

        parts = [f"Trích đoạn {name}: {summary}"]
        if play:
            parts.append(f"Thuộc vở: {play}.")
        if characters:
            parts.append(f"Các nhân vật xuất hiện: {', '.join(characters)}.")
        if actors:
            parts.append(f"Các diễn viên tham gia: {', '.join(actors)}.")

        chunks.append({
            "text": " ".join(parts),
            "metadata": {"type": NodeType.SCENE.lower(), "name": name},
        })
    logger.info("Scenes: %d chunks", len(chunks))
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
