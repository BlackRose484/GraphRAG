"""
Centralized logging setup for GraphRAGv2.

Features:
- Rotating file handler  → logs/graphrag.log  (10 MB / 5 backups)
- Daily archive handler  → logs/archive/graphrag-YYYY-MM-DD.log
- Console handler        → stderr, color-coded by level
- Log level configurable via env var LOG_LEVEL (default INFO)
- Single call: get_logger(__name__) in any module

Usage::

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Loading ontology...")
    logger.warning("Neo4j password missing")
    logger.error("LLM call failed", exc_info=True)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOG_DIR       = _PROJECT_ROOT / "logs"
_ARCHIVE_DIR   = _LOG_DIR / "archive"
_LOG_FILE      = _LOG_DIR / "graphrag.log"

_LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_TO_FILE   = os.getenv("LOG_TO_FILE", "true").lower() != "false"
_LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "true").lower() != "false"

# ── Formats ───────────────────────────────────────────────────────────────────

_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_CONSOLE_DATE_FMT = "%H:%M:%S"


# ── ANSI color wrapper for console ────────────────────────────────────────────

_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Add ANSI colors to level name in console output."""

    def format(self, record: logging.LogRecord) -> str:
        color  = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


# ── Root logger setup (called once at import) ─────────────────────────────────

def _setup_root_logger() -> None:
    """Configure the root 'graphrag' logger. Called once at module import."""
    root = logging.getLogger("graphrag")

    # Avoid adding duplicate handlers if already configured
    if root.handlers:
        return

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    root.propagate = False

    if _LOG_TO_FILE:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        # ── Rotating by size (main log) ────────────────────────────────────
        rotate_handler = RotatingFileHandler(
            filename=_LOG_FILE,
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        rotate_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FMT)
        )
        root.addHandler(rotate_handler)

        # ── Daily archive ──────────────────────────────────────────────────
        daily_handler = TimedRotatingFileHandler(
            filename=str(_ARCHIVE_DIR / "graphrag.log"),
            when="midnight",
            backupCount=30,
            encoding="utf-8",
        )
        daily_handler.suffix = "%Y-%m-%d.log"
        daily_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FMT)
        )
        root.addHandler(daily_handler)

    if _LOG_TO_STDOUT:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            _ColorFormatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATE_FMT)
        )
        root.addHandler(console_handler)


_setup_root_logger()

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'graphrag' hierarchy.

    Args:
        name: Typically ``__name__`` of the calling module.
              e.g. 'src.core.settings' → logger name 'graphrag.src.core.settings'

    Returns:
        A configured :class:`logging.Logger` instance.

    Example::

        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Ready")
    """
    # Strip project-relative prefix so names stay readable in log files
    clean = name.removeprefix("src.").removeprefix("GraphRAGv2.")
    return logging.getLogger(f"graphrag.{clean}")
