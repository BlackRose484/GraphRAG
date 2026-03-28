"""
CommunityIndex — Tự động phát hiện cộng đồng từ Knowledge Graph.

Tải TOÀN BỘ đồ thị từ Neo4j (tất cả 7 loại node + 7 loại quan hệ),
xây dựng networkx Graph trung thực, chạy community detection
(Greedy Modularity), rồi đặt tên cộng đồng từ Play node bên trong.

NGUYÊN TẮC: Không bỏ qua, không gộp, không tạo cạnh tổng hợp.
Mọi node và quan hệ đều phản ánh đúng cấu trúc trong Neo4j.

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

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from src.constants.constant import NodeProp, NodeType, RelType
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_FALLBACK_FILE = (
    Path(__file__).resolve().parent.parent.parent / "data" / "community_cache.json"
)

# ── Cypher: tải toàn bộ đồ thị trung thực ────────────────────────────────────

_LOAD_ALL_NODES_CYPHER = """
MATCH (n)
RETURN
    elementId(n)  AS node_id,
    labels(n)     AS labels,
    properties(n) AS props
"""

_LOAD_ALL_EDGES_CYPHER = """
MATCH (n)-[r]->(m)
RETURN
    elementId(n) AS from_id,
    elementId(m) AS to_id,
    type(r)      AS rel_type
"""

# Cypher phụ: lấy thông tin vai diễn để xây dựng role_assignments
# (vì RoleAssignment không có tên trực tiếp — cần join với Character và Actor)
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
    """Một cộng đồng được phát hiện tự động từ đồ thị."""

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
        lines: list[str] = [f"=== COMMUNITY [{self.community_id}]: {self.play_title} ==="]
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
    """Tự động phát hiện và lập chỉ mục cộng đồng từ Knowledge Graph."""

    def __init__(self) -> None:
        self._communities: dict[str, CommunitySubgraph] = {}
        self._entity_to_plays: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._source: str = "unloaded"

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, client: Any) -> None:
        """Tải toàn bộ đồ thị từ Neo4j, phát hiện cộng đồng."""
        try:
            self._load_and_detect(client)
            self._source = "Neo4j+detection"
        except Exception as exc:  # noqa: BLE001
            _logger.warning("CommunityIndex: detection failed (%s) — trying cache", exc)
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

    # ── Private — Load + Detect ───────────────────────────────────────────────

    def _load_and_detect(self, client: Any) -> None:
        """Tải toàn bộ graph → networkx → community detection."""

        # ── Bước 1: Tải tất cả node ───────────────────────────────────────────
        node_rows = client.read(_LOAD_ALL_NODES_CYPHER)
        _logger.info("CommunityIndex: %d nodes loaded", len(node_rows))

        node_meta: dict[str, dict[str, Any]] = {}  # elementId → {label, props}
        for row in node_rows:
            nid   = row["node_id"]
            lbls  = row.get("labels", [])
            props = row.get("props", {}) or {}
            label = lbls[0] if lbls else "Unknown"
            node_meta[nid] = {"label": label, "props": props}

        # ── Bước 2: Tải tất cả cạnh ────────────────────────────────────────────
        edge_rows = client.read(_LOAD_ALL_EDGES_CYPHER)
        _logger.info("CommunityIndex: %d edges loaded", len(edge_rows))

        # ── Bước 3: Tải role assignments để xây dựng CommunitySubgraph ────────
        role_rows = client.read(_LOAD_ROLE_ASSIGNMENTS_CYPHER)
        _logger.info("CommunityIndex: %d role assignments loaded", len(role_rows))

        # Lookup: ra_id → {char_name, actor_name, play_title, scene_name}
        role_by_id: dict[str, dict[str, Any]] = {
            row["ra_id"]: row for row in role_rows if row.get("ra_id")
        }

        # ── Bước 4: Xây dựng networkx graph trung thực ────────────────────────
        G = nx.Graph()
        for nid, meta in node_meta.items():
            G.add_node(nid, label=meta["label"], props=meta["props"])
        for row in edge_rows:
            fid, tid = row["from_id"], row["to_id"]
            if fid in node_meta and tid in node_meta:
                G.add_edge(fid, tid, rel=row["rel_type"])

        _logger.info(
            "CommunityIndex: networkx graph — %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )

        # ── Bước 5: Community detection ────────────────────────────────────────
        components = list(nx.connected_components(G))
        _logger.info("CommunityIndex: %d connected components", len(components))

        detected: list[frozenset[str]] = []
        for comp in components:
            subG = G.subgraph(comp)
            if subG.number_of_nodes() < 3:
                detected.append(frozenset(comp))
            else:
                detected.extend(greedy_modularity_communities(subG))

        _logger.info("CommunityIndex: %d communities detected", len(detected))

        # ── Bước 6: Xây dựng CommunitySubgraph từ mỗi cụm ────────────────────
        for comm_id, node_set in enumerate(detected):
            comm = self._build_community(comm_id, node_set, node_meta, G, role_by_id)
            if comm:
                self._communities[comm.play_title] = comm

        if self._communities:
            try:
                self.save_cache()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("CommunityIndex: cache write failed: %s", exc)

    def _build_community(
        self,
        comm_id: int,
        node_set: frozenset[str],
        node_meta: dict[str, dict[str, Any]],
        G: nx.Graph,
        role_by_id: dict[str, dict[str, Any]],
    ) -> CommunitySubgraph | None:
        """Xây dựng CommunitySubgraph từ tập node đã phát hiện."""
        characters:  list[dict[str, Any]] = []
        actors:      list[dict[str, Any]] = []
        scenes:      list[dict[str, Any]] = []
        play_titles: list[str] = []

        # Phân loại node theo label
        for nid in node_set:
            meta  = node_meta.get(nid, {})
            label = meta.get("label", "")
            props = meta.get("props", {}) or {}

            if label == NodeType.PLAY:
                title = props.get("title", "")
                if title:
                    play_titles.append(title)

            elif label == NodeType.CHARACTER:
                name = props.get("charName", "")
                if name:
                    characters.append({"name": name, "gender": props.get("charGender", "")})

            elif label == NodeType.ACTOR:
                name = props.get("actorName", "")
                if name:
                    actors.append({"name": name})

            elif label == NodeType.SCENE:
                name = props.get("sceneName", "")
                if name:
                    scenes.append({"name": name, "summary": props.get("sceneSummary", "")})

        # Cộng đồng không có thực thể có nghĩa → bỏ qua
        if not characters and not actors and not scenes:
            return None

        # Đặt tên cộng đồng từ Play node
        if play_titles:
            community_name = " & ".join(sorted(play_titles))
        else:
            # Không có Play node → dùng nút bậc cao nhất
            subG = G.subgraph(node_set)
            top_nid = max(node_set, key=lambda n: subG.degree(n), default=None)
            meta = node_meta.get(top_nid, {}) if top_nid else {}
            props = meta.get("props", {}) or {}
            community_name = (
                props.get("charName")
                or props.get("actorName")
                or props.get("sceneName")
                or f"Community_{comm_id}"
            )

        # Xây dựng role_assignments: chỉ lấy RoleAssignment thuộc cụm này
        role_assignments: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for nid in node_set:
            meta = node_meta.get(nid, {})
            if meta.get("label") == NodeType.ROLE_ASSIGNMENT:
                info = role_by_id.get(nid, {})
                char  = info.get("char_name", "")
                actor = info.get("actor_name", "")
                if char and actor and (char, actor) not in seen_pairs:
                    seen_pairs.add((char, actor))
                    role_assignments.append({
                        "character": char,
                        "actor":     actor,
                        "scene":     info.get("scene_name", ""),
                        "play":      info.get("play_title", ""),
                        "version":   info.get("version_id", ""),
                    })

        return CommunitySubgraph(
            community_id=comm_id,
            play_title=community_name,
            characters=characters,
            actors=actors,
            scenes=scenes,
            role_assignments=role_assignments,
        )

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
