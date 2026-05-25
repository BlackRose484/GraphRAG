"""Dynamic entity list pulled from Neo4j at startup, with file fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_FALLBACK_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "cheo_entities.txt"

_CYPHER: dict[str, str] = {
    "plays":      "MATCH (p:Play)      RETURN p.title     AS name ORDER BY name",
    "scenes":     "MATCH (s:Scene)     RETURN s.sceneName AS name ORDER BY name",
    "characters": "MATCH (c:Character) RETURN c.charName  AS name ORDER BY name",
    "actors":     "MATCH (a:Actor)     RETURN a.actorName AS name ORDER BY name",
}

_HEADERS: dict[str, str] = {
    "plays":      "VỞ CHÈO",
    "scenes":     "TRÍCH ĐOẠN",
    "characters": "NHÂN VẬT",
    "actors":     "DIỄN VIÊN",
}


class EntityCatalog:
    """Cached entity catalog built from Neo4j (or fallback file)."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = {k: [] for k in _CYPHER}
        self._text: str = ""
        self._loaded: bool = False
        self._source: str = "unloaded"

    def load(self, client: Any) -> None:
        """Pull entity names from Neo4j and build the catalog."""
        try:
            self._load_from_neo4j(client)
            self._source = "Neo4j"
        except Exception as exc:
            _logger.warning(
                "EntityCatalog: Neo4j load failed (%s) — falling back to %s",
                exc, _FALLBACK_FILE,
            )
            self._load_from_file()
            self._source = f"fallback:{_FALLBACK_FILE.name}"

        self._text = self._build_text()
        self._loaded = True
        total = sum(len(v) for v in self._data.values())
        _logger.info(
            "EntityCatalog loaded %d entities from %s "
            "(plays=%d, scenes=%d, characters=%d, actors=%d)",
            total, self._source,
            len(self._data["plays"]),
            len(self._data["scenes"]),
            len(self._data["characters"]),
            len(self._data["actors"]),
        )

    def refresh(self, client: Any) -> None:
        """Re-load the catalog (call after ontology updates)."""
        _logger.info("EntityCatalog: refreshing from %s", self._source)
        self.load(client)

    def as_text(self) -> str:
        """Return the formatted catalog string for prompt injection."""
        if not self._loaded:
            _logger.warning("EntityCatalog.as_text() called before load() — returning empty")
            return ""
        return self._text

    def as_dict(self) -> dict[str, list[str]]:
        """Return raw entity lists by category."""
        return dict(self._data)

    def is_loaded(self) -> bool:
        return self._loaded

    def _load_from_neo4j(self, client: Any) -> None:
        for key, cypher in _CYPHER.items():
            rows = client.read(cypher)
            names = [
                str(row["name"]).strip()
                for row in rows
                if row.get("name") and str(row["name"]).strip()
            ]
            self._data[key] = names
            _logger.debug("EntityCatalog: %s — %d names from Neo4j", key, len(names))

    def _load_from_file(self) -> None:
        if not _FALLBACK_FILE.exists():
            _logger.error("EntityCatalog: fallback file not found: %s", _FALLBACK_FILE)
            return

        key_map = {
            "VỞ CHÈO": "plays",
            "TRÍCH ĐOẠN": "scenes",
            "NHÂN VẬT": "characters",
            "DIỄN VIÊN": "actors",
        }
        current_key: str | None = None

        with open(_FALLBACK_FILE, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                header = line.rstrip(":")
                if header in key_map:
                    current_key = key_map[header]
                    continue
                if current_key:
                    names = [n.strip() for n in line.split(",") if n.strip()]
                    self._data[current_key].extend(names)

        for key, names in self._data.items():
            _logger.debug("EntityCatalog fallback: %s — %d names", key, len(names))

    def _build_text(self) -> str:
        parts: list[str] = []
        for key, header in _HEADERS.items():
            names = self._data.get(key, [])
            if names:
                parts.append(f"{header}:\n{', '.join(names)}")
        return "\n\n".join(parts)
