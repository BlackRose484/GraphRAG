"""
CommunityIndex — Play-centric community subgraphs, pre-loaded at startup.

Each **Play** in the Cheo Knowledge Graph forms a natural community: the set
of Characters, Actors, Scenes, RoleAssignments, and Versions that belong to
that play.  This module loads all communities once at startup and provides:

* O(1) entity → play(s) lookup via a reverse index
* Pre-built subgraph data per play (no Cypher at query time)
* Formatted context text for LLM prompt injection

Usage::

    from src.g_retrieval.community_index import CommunityIndex

    idx = CommunityIndex()
    idx.load(neo4j_client)                    # call once at startup

    communities = idx.resolve("Thị Kính")     # → [CommunitySubgraph(...)]
    context     = idx.as_context(["Quan Âm Thị Kính"])  # formatted text
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.constants.constant import NodeProp, NodeType, RelType
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Fallback cache if Neo4j is unavailable
_FALLBACK_FILE = (
    Path(__file__).resolve().parent.parent.parent / "data" / "community_cache.json"
)

# ── Cypher: load one play's full community ────────────────────────────────────

_COMMUNITY_CYPHER = """
MATCH (p:Play {title: $play_title})
OPTIONAL MATCH (p)-[:HAS_CHARACTER]->(c:Character)
OPTIONAL MATCH (p)-[:HAS_SCENE]->(s:Scene)
OPTIONAL MATCH (c)<-[:FOR_CHARACTER]-(ra:RoleAssignment)-[:PERFORMED_BY]->(a:Actor)
OPTIONAL MATCH (ra)-[:IN_VERSION]->(v:Version)
RETURN p.title AS play_title,
       collect(DISTINCT {name: c.charName, gender: c.charGender})  AS characters,
       collect(DISTINCT {name: s.sceneName, summary: s.sceneSummary}) AS scenes,
       collect(DISTINCT {name: a.actorName}) AS actors,
       collect(DISTINCT {
           character: c.charName,
           actor:     a.actorName,
           version:   v.versionId
       }) AS role_assignments
"""

_ALL_PLAYS_CYPHER = "MATCH (p:Play) RETURN p.title AS title ORDER BY title"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class CommunitySubgraph:
    """One play's complete community subgraph."""

    play_title: str
    characters: list[dict[str, Any]] = field(default_factory=list)
    actors: list[dict[str, Any]] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)
    role_assignments: list[dict[str, Any]] = field(default_factory=list)

    # ── Derived sets for fast lookup ──────────────────────────────────────

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
        """All searchable names in this community."""
        return self.character_names | self.actor_names | self.scene_names | {self.play_title}

    def summary_line(self) -> str:
        return (
            f"{self.play_title}: "
            f"{len(self.character_names)} nhân vật, "
            f"{len(self.actor_names)} diễn viên, "
            f"{len(self.scene_names)} trích đoạn, "
            f"{len(self.role_assignments)} vai diễn"
        )

    def as_text(self) -> str:
        """Format this community as a readable text block for LLM context."""
        lines: list[str] = [f"=== COMMUNITY: {self.play_title} ==="]

        if self.character_names:
            chars = sorted(self.character_names)
            lines.append(f"Nhân vật ({len(chars)}): {', '.join(chars)}")

        if self.actor_names:
            actors = sorted(self.actor_names)
            lines.append(f"Diễn viên ({len(actors)}): {', '.join(actors)}")

        if self.scene_names:
            scenes = sorted(self.scene_names)
            lines.append(f"Trích đoạn ({len(scenes)}): {', '.join(scenes)}")

        # Role assignment details
        valid_roles = [
            r for r in self.role_assignments
            if r.get("character") and r.get("actor")
        ]
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
        """Serializable dict for cache file."""
        return {
            "play_title": self.play_title,
            "characters": self.characters,
            "actors": self.actors,
            "scenes": self.scenes,
            "role_assignments": self.role_assignments,
        }


# ── Main index ────────────────────────────────────────────────────────────────


class CommunityIndex:
    """Pre-loaded index of all play-centric communities."""

    def __init__(self) -> None:
        self._communities: dict[str, CommunitySubgraph] = {}
        self._entity_to_plays: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._source: str = "unloaded"

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, client: Any) -> None:
        """Load all play communities from Neo4j (fallback to cache file).

        Args:
            client: Connected :class:`~src.graph_loader.neo4j_client.Neo4jClient`.
        """
        try:
            self._load_from_neo4j(client)
            self._source = "Neo4j"
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "CommunityIndex: Neo4j load failed (%s) — trying cache", exc,
            )
            self._load_from_cache()
            self._source = f"cache:{_FALLBACK_FILE.name}"

        self._build_reverse_index()
        self._loaded = True
        _logger.info(
            "CommunityIndex loaded %d communities from %s:\n%s",
            len(self._communities),
            self._source,
            "\n".join(f"  {c.summary_line()}" for c in self._communities.values()),
        )

    def resolve(self, entity_name: str) -> list[CommunitySubgraph]:
        """Return all community subgraphs containing *entity_name*.

        Uses case-insensitive matching against the reverse index.

        Args:
            entity_name: Any entity name (character, actor, scene, or play).

        Returns:
            List of :class:`CommunitySubgraph` (may be empty).
        """
        if not self._loaded:
            _logger.warning("CommunityIndex.resolve() called before load()")
            return []
        key = entity_name.strip().lower()
        play_titles = self._entity_to_plays.get(key, [])
        return [self._communities[t] for t in play_titles if t in self._communities]

    def resolve_many(self, entity_names: list[str]) -> list[CommunitySubgraph]:
        """Resolve multiple entity names and return deduplicated communities."""
        seen_titles: set[str] = set()
        result: list[CommunitySubgraph] = []
        for name in entity_names:
            for comm in self.resolve(name):
                if comm.play_title not in seen_titles:
                    seen_titles.add(comm.play_title)
                    result.append(comm)
        return result

    def get_community(self, play_title: str) -> CommunitySubgraph | None:
        """Retrieve a specific community by play title."""
        return self._communities.get(play_title)

    def all_plays(self) -> list[str]:
        """Return sorted list of all play titles."""
        return sorted(self._communities.keys())

    def as_context(self, play_titles: list[str]) -> str:
        """Format selected communities as a single context string for LLM.

        Args:
            play_titles: List of play titles to include.

        Returns:
            Combined formatted text of all matched communities.
        """
        parts: list[str] = []
        for title in play_titles:
            comm = self._communities.get(title)
            if comm:
                parts.append(comm.as_text())
        return "\n\n".join(parts) if parts else ""

    def as_graph_data(self, play_titles: list[str]) -> dict[str, Any]:
        """Convert community subgraphs to GraphData-compatible dict.

        This allows community data to flow through the existing
        GraphFormatConverter pipeline.

        Args:
            play_titles: Which communities to include.

        Returns:
            Dict with ``nodes``, ``triplets``, ``community_context`` keys.
        """
        nodes: list[dict[str, Any]] = []
        triplets: list[tuple[str, str, str]] = []
        seen_nodes: set[str] = set()
        seen_triplets: set[tuple[str, str, str]] = set()

        for title in play_titles:
            comm = self._communities.get(title)
            if not comm:
                continue

            # Characters as nodes
            for c in comm.characters:
                name = c.get("name", "")
                if name and name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({
                        NodeProp.CHAR_NAME: name,
                        NodeProp.CHAR_GENDER: c.get("gender", ""),
                    })

            # Actors as nodes
            for a in comm.actors:
                name = a.get("name", "")
                if name and name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({NodeProp.ACTOR_NAME: name})

            # Scenes as nodes
            for s in comm.scenes:
                name = s.get("name", "")
                if name and name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({
                        NodeProp.SCENE_NAME: name,
                        NodeProp.SCENE_SUMMARY: s.get("summary", ""),
                    })

            # Role assignments → triplets
            for r in comm.role_assignments:
                char = r.get("character", "")
                actor = r.get("actor", "")
                if char and actor:
                    t = (char, RelType.PERFORMED_BY, actor)
                    if t not in seen_triplets:
                        seen_triplets.add(t)
                        triplets.append(t)

            # Play → Character triplets
            for c_name in comm.character_names:
                t = (title, RelType.HAS_CHARACTER, c_name)
                if t not in seen_triplets:
                    seen_triplets.add(t)
                    triplets.append(t)

            # Play → Scene triplets
            for s_name in comm.scene_names:
                t = (title, RelType.HAS_SCENE, s_name)
                if t not in seen_triplets:
                    seen_triplets.add(t)
                    triplets.append(t)

        return {
            "nodes": nodes,
            "triplets": triplets,
            "community_context": self.as_context(play_titles),
        }

    def is_loaded(self) -> bool:
        return self._loaded

    def save_cache(self) -> None:
        """Write current communities to fallback cache file."""
        data = {t: c.to_dict() for t, c in self._communities.items()}
        _FALLBACK_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        _logger.info("CommunityIndex: cache saved to %s", _FALLBACK_FILE)

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_from_neo4j(self, client: Any) -> None:
        """Load every Play community via Cypher."""
        # Step 1: get all play titles
        play_rows = client.read(_ALL_PLAYS_CYPHER)
        titles = [r["title"] for r in play_rows if r.get("title")]
        _logger.info("CommunityIndex: found %d plays in Neo4j", len(titles))

        # Step 2: load each community
        for title in titles:
            rows = client.read(_COMMUNITY_CYPHER, {"play_title": title})
            if not rows:
                _logger.warning("CommunityIndex: no data for play '%s'", title)
                continue

            row = rows[0]
            comm = CommunitySubgraph(
                play_title=title,
                characters=[
                    c for c in row.get("characters", [])
                    if c and c.get("name")
                ],
                actors=[
                    a for a in row.get("actors", [])
                    if a and a.get("name")
                ],
                scenes=[
                    s for s in row.get("scenes", [])
                    if s and s.get("name")
                ],
                role_assignments=[
                    r for r in row.get("role_assignments", [])
                    if r and r.get("character") and r.get("actor")
                ],
            )
            self._communities[title] = comm
            _logger.debug("  loaded: %s", comm.summary_line())

        # Save cache for fallback
        if self._communities:
            try:
                self.save_cache()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("CommunityIndex: cache save failed: %s", exc)

    def _load_from_cache(self) -> None:
        """Load from fallback JSON cache file."""
        if not _FALLBACK_FILE.exists():
            _logger.error("CommunityIndex: cache file not found: %s", _FALLBACK_FILE)
            return

        raw = json.loads(_FALLBACK_FILE.read_text(encoding="utf-8"))
        for title, data in raw.items():
            self._communities[title] = CommunitySubgraph(
                play_title=data.get("play_title", title),
                characters=data.get("characters", []),
                actors=data.get("actors", []),
                scenes=data.get("scenes", []),
                role_assignments=data.get("role_assignments", []),
            )
        _logger.info("CommunityIndex: loaded %d communities from cache", len(self._communities))

    def _build_reverse_index(self) -> None:
        """Build entity_name (lowercase) → [play_title, ...] mapping."""
        self._entity_to_plays.clear()
        for title, comm in self._communities.items():
            for name in comm.all_entity_names:
                key = name.strip().lower()
                if key not in self._entity_to_plays:
                    self._entity_to_plays[key] = []
                if title not in self._entity_to_plays[key]:
                    self._entity_to_plays[key].append(title)
        _logger.debug(
            "CommunityIndex: reverse index has %d entries",
            len(self._entity_to_plays),
        )
