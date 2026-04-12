"""
CommunityIndex — Nhóm tri thức theo vở chèo từ Knowledge Graph.

Thay vì dùng thuật toán community detection tổng quát (greedy modularity),
index này nhóm trực tiếp theo Play node qua ba Cypher query:
    Play ──[HAS_CHARACTER]──► Character
    Play ──[HAS_SCENE]──────► Scene
    Play ◄── Scene ◄── Version ◄── RoleAssignment ──► Actor

Đảm bảo: toàn bộ nhân vật và diễn viên của mỗi vở luôn nằm trong
cùng một community, tránh hiện tượng phân mảnh dữ liệu.

Cấu trúc đồ thị thực tế:
    Play ──[HAS_CHARACTER]──► Character
    Play ──[HAS_SCENE]──────► Scene
    Scene ──[HAS_VERSION]──► Version
    RoleAssignment ──[FOR_CHARACTER]──► Character
    RoleAssignment ──[PERFORMED_BY]──► Actor
    RoleAssignment ──[IN_VERSION]──► Version
    RoleAssignment ──[HAS_APPEARANCE]──► Appearance
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.constants.constant import NodeProp, NodeType, RelType
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_FALLBACK_FILE = (
    Path(__file__).resolve().parent.parent.parent / "data" / "community_cache.json"
)

# ── Cypher queries ────────────────────────────────────────────────────────────

_LOAD_ALL_NODES_CYPHER = """
MATCH (n)
RETURN
    elementId(n)  AS node_id,
    labels(n)     AS labels,
    properties(n) AS props
"""

# Quan hệ trực tiếp Play → Character
_LOAD_PLAY_CHARACTERS_CYPHER = """
MATCH (p:Play)-[:HAS_CHARACTER]->(c:Character)
RETURN p.title AS play_title, c.charName AS char_name, c.charGender AS char_gender
"""

# Quan hệ trực tiếp Play → Scene
_LOAD_PLAY_SCENES_CYPHER = """
MATCH (p:Play)-[:HAS_SCENE]->(s:Scene)
RETURN p.title AS play_title, s.sceneName AS scene_name
"""

# Vai diễn đầy đủ: actor + character + play + scene qua RoleAssignment
_LOAD_ROLE_ASSIGNMENTS_CYPHER = """
MATCH (ra:RoleAssignment)
OPTIONAL MATCH (ra)-[:FOR_CHARACTER]->(c:Character)
OPTIONAL MATCH (ra)-[:PERFORMED_BY]->(a:Actor)
OPTIONAL MATCH (ra)-[:IN_VERSION]->(v:Version)<-[:HAS_VERSION]-(s:Scene)<-[:HAS_SCENE]-(p:Play)
RETURN
    elementId(ra) AS ra_id,
    c.charName    AS char_name,
    a.actorName   AS actor_name,
    v.versionId   AS version_id,
    s.sceneName   AS scene_name,
    p.title       AS play_title
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CommunitySubgraph:
    """Toàn bộ tri thức của một vở chèo."""

    community_id: int
    play_title: str
    characters: list[dict[str, Any]] = field(default_factory=list)
    actors: list[dict[str, Any]] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)
    role_assignments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def character_names(self) -> set[str]:
        return {c["name"] for c in self.characters if c.get("name")}

    @property
    def actor_names(self) -> set[str]:
        return {a["name"] for a in self.actors if a.get("name")}

    @property
    def scene_names(self) -> set[str]:
        return {s["name"] for s in self.scenes if s.get("name")}

    @property
    def all_entity_names(self) -> set[str]:
        return self.character_names | self.actor_names | self.scene_names | {self.play_title}

    def summary_line(self) -> str:
        return (
            f"[{self.community_id}] {self.play_title}: "
            f"{len(self.character_names)} nhân vật, "
            f"{len(self.actor_names)} diễn viên, "
            f"{len(self.scene_names)} trích đoạn, "
            f"{len(self.role_assignments)} vai diễn"
        )

    def as_text(self) -> str:
        lines: list[str] = [f"=== {self.play_title} ==="]
        if self.character_names:
            lines.append(f"Nhân vật ({len(self.character_names)}): {', '.join(sorted(self.character_names))}")
        if self.actor_names:
            lines.append(f"Diễn viên ({len(self.actor_names)}): {', '.join(sorted(self.actor_names))}")
        if self.scene_names:
            lines.append(f"Trích đoạn ({len(self.scene_names)}): {', '.join(sorted(self.scene_names))}")
        valid_roles = [r for r in self.role_assignments if r.get("character") and r.get("actor")]
        if valid_roles:
            lines.append("Quan hệ diễn xuất:")
            seen: set[tuple[str, str]] = set()
            for r in valid_roles:
                pair = (r["actor"], r["character"])
                if pair not in seen:
                    seen.add(pair)
                    lines.append(f"  • {r['actor']} → vai {r['character']}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "play_title": self.play_title,
            "characters": self.characters,
            "actors": self.actors,
            "scenes": self.scenes,
            "role_assignments": self.role_assignments,
        }


# ── Main class ────────────────────────────────────────────────────────────────

class CommunityIndex:
    """Lập chỉ mục tri thức theo từng vở chèo từ Knowledge Graph."""

    def __init__(self) -> None:
        self._communities: dict[str, CommunitySubgraph] = {}
        self._entity_to_plays: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._source: str = "unloaded"

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, client: Any) -> None:
        """Tải toàn bộ đồ thị từ Neo4j, nhóm theo vở chèo."""
        try:
            self._load_and_detect(client)
            self._source = "Neo4j+play-centric"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("CommunityIndex: load failed (%s) — trying cache", exc)
            self._load_from_cache()
            self._source = f"cache:{_FALLBACK_FILE.name}"

        self._build_reverse_index()
        self._loaded = True
        _logger.info(
            "CommunityIndex: %d communities from %s:\n%s",
            len(self._communities), self._source,
            "\n".join(f"  {c.summary_line()}" for c in self._communities.values()),
        )

    def resolve(self, entity_name: str) -> list[CommunitySubgraph]:
        if not self._loaded:
            return []
        key = entity_name.strip().lower()
        return [self._communities[t] for t in self._entity_to_plays.get(key, []) if t in self._communities]

    def resolve_many(self, entity_names: list[str]) -> list[CommunitySubgraph]:
        seen: set[str] = set()
        result: list[CommunitySubgraph] = []
        for name in entity_names:
            for comm in self.resolve(name):
                if comm.play_title not in seen:
                    seen.add(comm.play_title)
                    result.append(comm)
        return result

    def get_community(self, play_title: str) -> CommunitySubgraph | None:
        return self._communities.get(play_title)

    def all_plays(self) -> list[str]:
        return sorted(self._communities.keys())

    def as_context(self, play_titles: list[str]) -> str:
        parts = [self._communities[t].as_text() for t in play_titles if t in self._communities]
        return "\n\n".join(parts)

    def as_graph_data(self, play_titles: list[str]) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        triplets: list[tuple[str, str, str]] = []
        seen_nodes: set[str] = set()
        seen_triplets: set[tuple[str, str, str]] = set()

        for title in play_titles:
            comm = self._communities.get(title)
            if not comm:
                continue
            for c in comm.characters:
                name = c.get("name", "")
                if name and name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({NodeProp.CHAR_NAME: name, NodeProp.CHAR_GENDER: c.get("gender", "")})
            for a in comm.actors:
                name = a.get("name", "")
                if name and name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({NodeProp.ACTOR_NAME: name})
            for s in comm.scenes:
                name = s.get("name", "")
                if name and name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({NodeProp.SCENE_NAME: name})
            for r in comm.role_assignments:
                char, actor = r.get("character", ""), r.get("actor", "")
                if char and actor:
                    t = (char, RelType.PERFORMED_BY, actor)
                    if t not in seen_triplets:
                        seen_triplets.add(t)
                        triplets.append(t)
            for c_name in comm.character_names:
                t = (title, RelType.HAS_CHARACTER, c_name)
                if t not in seen_triplets:
                    seen_triplets.add(t)
                    triplets.append(t)
            for s_name in comm.scene_names:
                t = (title, RelType.HAS_SCENE, s_name)
                if t not in seen_triplets:
                    seen_triplets.add(t)
                    triplets.append(t)

        return {"nodes": nodes, "triplets": triplets, "community_context": self.as_context(play_titles)}

    def is_loaded(self) -> bool:
        return self._loaded

    def save_cache(self) -> None:
        data = {t: c.to_dict() for t, c in self._communities.items()}
        _FALLBACK_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _logger.info("CommunityIndex: cache saved → %s", _FALLBACK_FILE)

    # ── Private — Load ────────────────────────────────────────────────────────

    def _load_and_detect(self, client: Any) -> None:
        """Tải graph từ Neo4j và nhóm trực tiếp theo vở chèo.

        Dùng 3 Cypher query để lấy đầy đủ dữ liệu từng vở:
            1. Play → Character  (quan hệ trực tiếp)
            2. Play → Scene      (quan hệ trực tiếp)
            3. RoleAssignment    (actor + character + play + scene qua path 4 hop)

        Không dùng community detection algorithm để tránh phân mảnh dữ liệu.
        """
        # Load node count for logging
        node_rows = client.read(_LOAD_ALL_NODES_CYPHER)
        _logger.info("CommunityIndex: %d nodes in graph", len(node_rows))

        # Load direct Play→Character links
        char_rows = client.read(_LOAD_PLAY_CHARACTERS_CYPHER)
        _logger.info("CommunityIndex: %d play-character links", len(char_rows))

        # Load direct Play→Scene links
        scene_rows = client.read(_LOAD_PLAY_SCENES_CYPHER)
        _logger.info("CommunityIndex: %d play-scene links", len(scene_rows))

        # Load role assignments (actor + character + play + scene)
        role_rows = client.read(_LOAD_ROLE_ASSIGNMENTS_CYPHER)
        _logger.info("CommunityIndex: %d role assignments", len(role_rows))

        # ── Group by play title ───────────────────────────────────────────────
        play_characters: dict[str, list[dict[str, Any]]] = {}
        play_actors:     dict[str, list[dict[str, Any]]] = {}
        play_scenes:     dict[str, list[dict[str, Any]]] = {}
        play_roles:      dict[str, list[dict[str, Any]]] = {}

        seen_chars:  dict[str, set[str]]              = {}
        seen_actors: dict[str, set[str]]              = {}
        seen_scenes: dict[str, set[str]]              = {}
        seen_pairs:  dict[str, set[tuple[str, str]]]  = {}

        for row in char_rows:
            play, char = row.get("play_title", ""), row.get("char_name", "")
            if play and char and char not in seen_chars.setdefault(play, set()):
                seen_chars[play].add(char)
                play_characters.setdefault(play, []).append(
                    {"name": char, "gender": row.get("char_gender", "")}
                )

        for row in scene_rows:
            play, scene = row.get("play_title", ""), row.get("scene_name", "")
            if play and scene and scene not in seen_scenes.setdefault(play, set()):
                seen_scenes[play].add(scene)
                play_scenes.setdefault(play, []).append({"name": scene})

        for row in role_rows:
            play  = row.get("play_title", "")
            actor = row.get("actor_name", "")
            char  = row.get("char_name", "")
            scene = row.get("scene_name", "") or ""
            if not play:
                continue
            if actor and actor not in seen_actors.setdefault(play, set()):
                seen_actors[play].add(actor)
                play_actors.setdefault(play, []).append({"name": actor})
            if char and actor:
                pair = (char, actor)
                if pair not in seen_pairs.setdefault(play, set()):
                    seen_pairs[play].add(pair)
                    play_roles.setdefault(play, []).append({
                        "character": char,
                        "actor":     actor,
                        "scene":     scene,
                        "play":      play,
                        "version":   row.get("version_id", ""),
                    })

        # ── Build one CommunitySubgraph per play ──────────────────────────────
        all_plays = sorted(
            set(play_characters) | set(play_actors) | set(play_scenes) | set(play_roles)
        )
        _logger.info("CommunityIndex: building %d play-centric communities", len(all_plays))

        for comm_id, play_title in enumerate(all_plays):
            self._communities[play_title] = CommunitySubgraph(
                community_id=comm_id,
                play_title=play_title,
                characters=play_characters.get(play_title, []),
                actors=play_actors.get(play_title, []),
                scenes=play_scenes.get(play_title, []),
                role_assignments=play_roles.get(play_title, []),
            )

        if self._communities:
            try:
                self.save_cache()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("CommunityIndex: cache write failed: %s", exc)

    # ── Private — Cache fallback ──────────────────────────────────────────────

    def _load_from_cache(self) -> None:
        if not _FALLBACK_FILE.exists():
            _logger.error("CommunityIndex: cache not found: %s", _FALLBACK_FILE)
            return
        raw = json.loads(_FALLBACK_FILE.read_text(encoding="utf-8"))
        for title, data in raw.items():
            self._communities[title] = CommunitySubgraph(
                community_id=data.get("community_id", 0),
                play_title=data.get("play_title", title),
                characters=data.get("characters", []),
                actors=data.get("actors", []),
                scenes=data.get("scenes", []),
                role_assignments=data.get("role_assignments", []),
            )
        _logger.info("CommunityIndex: %d communities from cache", len(self._communities))

    def _build_reverse_index(self) -> None:
        self._entity_to_plays.clear()
        for title, comm in self._communities.items():
            for name in comm.all_entity_names:
                key = name.strip().lower()
                if key:
                    self._entity_to_plays.setdefault(key, [])
                    if title not in self._entity_to_plays[key]:
                        self._entity_to_plays[key].append(title)
