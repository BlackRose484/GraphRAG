"""Demo cache — phục vụ quay video demo khi API LLM hết quota."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_CACHE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark" / "datasets" / "demo_cache.json"
)


def is_enabled() -> bool:
    """``True`` nếu env ``DEMO_MODE`` được bật (1/true/yes)."""
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def lookup(query: str) -> Optional[dict[str, Any]]:
    """Tìm câu hỏi trong cache (case-insensitive, bỏ dấu chấm câu cuối)."""
    if not query:
        return None
    key = _normalize(query)
    cache = _load_cache()
    return cache.get(key)


def all_questions() -> list[str]:
    return [v["_original_query"] for v in _load_cache().values()]


def cache_path() -> Path:
    return _CACHE_PATH


def reload() -> None:
    """Force re-read cache file (clear lru_cache)."""
    _load_cache.cache_clear()


@lru_cache(maxsize=1)
def _load_cache() -> dict[str, dict[str, Any]]:
    if not _CACHE_PATH.exists():
        _logger.info("Demo cache file không tồn tại: %s", _CACHE_PATH)
        return {}
    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _logger.error("Demo cache file lỗi JSON: %s", exc)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for q in raw.get("questions", []):
        query = q.get("query", "").strip()
        if not query:
            continue
        out[_normalize(query)] = {
            "_original_query": query,
            "answers":         q.get("answers", {}),
        }
    _logger.info("Demo cache loaded: %d câu hỏi từ %s", len(out), _CACHE_PATH.name)
    return out


_TRAIL_PUNCT = re.compile(r"[.!?。！？\s]+$")
_MULTI_WS    = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _TRAIL_PUNCT.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    return text
