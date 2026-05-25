"""HistoryStore — lưu lịch sử hội thoại vào file JSON local."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_HISTORY_FILE = Path(__file__).resolve().parents[3] / "data" / "chat_history.json"
_MAX_ENTRIES = 500  # cap để tránh file phình to


class HistoryStore:
    """Thread-safe (single-process) JSON-backed chat history."""

    @classmethod
    def append(
        cls,
        page: str,
        query: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "id":        str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "page":      page,
            "query":     query,
            "answer":    answer,
            "metadata":  metadata or {},
        }
        try:
            entries = cls._read_raw()
            entries.insert(0, entry)
            entries = entries[:_MAX_ENTRIES]
            cls._write_raw(entries)
        except Exception:
            pass  # never crash the main pipeline

    @classmethod
    def load(cls, page_filter: str | None = None, limit: int = 200) -> list[dict]:
        """Load history entries, newest first."""
        try:
            entries = cls._read_raw()
            if page_filter:
                entries = [e for e in entries if e.get("page") == page_filter]
            return entries[:limit]
        except Exception:
            return []

    @classmethod
    def count(cls) -> dict[str, int]:
        """Return entry counts per page and total."""
        try:
            entries = cls._read_raw()
            counts: dict[str, int] = {}
            for e in entries:
                p = e.get("page", "unknown")
                counts[p] = counts.get(p, 0) + 1
            counts["total"] = len(entries)
            return counts
        except Exception:
            return {"total": 0}

    @classmethod
    def clear(cls, page_filter: str | None = None) -> int:
        """Delete entries. Returns number deleted."""
        try:
            entries = cls._read_raw()
            before = len(entries)
            if page_filter:
                entries = [e for e in entries if e.get("page") != page_filter]
            else:
                entries = []
            cls._write_raw(entries)
            return before - len(entries)
        except Exception:
            return 0

    @classmethod
    def _read_raw(cls) -> list[dict]:
        if not _HISTORY_FILE.exists():
            return []
        with open(_HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    @classmethod
    def _write_raw(cls, entries: list[dict]) -> None:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
