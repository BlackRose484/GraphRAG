"""
Centralized logging setup for GraphRAGv2.

Log structure (per-run, per-day):
    logs/
      2026-02-28/
        run_14-30-45.log      ← mỗi lần chạy app/script = 1 file riêng
        run_15-12-03.log
      2026-03-01/
        run_09-00-12.log

Features:
- Per-run file handler  → logs/YYYY-MM-DD/run_HH-MM-SS.log
- Console handler       → stdout, color-coded by level
- Log level configurable via env var LOG_LEVEL  (default: INFO)
- LOG_TO_FILE / LOG_TO_STDOUT toggles via env

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
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_LOG_ROOT      = _PROJECT_ROOT / "logs"

_LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_TO_FILE   = os.getenv("LOG_TO_FILE",   "true").lower() != "false"
_LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "true").lower() != "false"

# Timestamp captured once at process start → shared by all loggers in this run
_RUN_START     = datetime.now()
_DATE_FOLDER   = _RUN_START.strftime("%Y-%m-%d")          # e.g. 2026-02-28
_RUN_FILENAME  = _RUN_START.strftime("run_%H-%M-%S.log")  # e.g. run_14-30-45.log
_RUN_LOG_FILE  = _LOG_ROOT / _DATE_FOLDER / _RUN_FILENAME

# ── Formats ───────────────────────────────────────────────────────────────────

_FILE_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATE_FMT  = "%H:%M:%S"

_CONSOLE_FORMAT   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
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
    """Attach ANSI color to the level name for console output."""

    def format(self, record: logging.LogRecord) -> str:
        # Work on a copy so we don't mutate the shared LogRecord
        record = logging.makeLogRecord(record.__dict__)
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


# ── Root logger setup (called once at module import) ─────────────────────────

def _setup_root_logger() -> None:
    root = logging.getLogger("graphrag")

    if root.handlers:           # already configured (e.g. Streamlit hot-reload)
        return

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    root.propagate = False

    # ── Per-run file handler ───────────────────────────────────────────────
    if _LOG_TO_FILE:
        _RUN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            filename=str(_RUN_LOG_FILE),
            mode="w",           # new file each run
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FMT)
        )
        root.addHandler(file_handler)

        # Write a header so it's easy to identify the run in a log viewer
        root.info(
            "══ Run started at %s ══  log → %s",
            _RUN_START.strftime("%Y-%m-%d %H:%M:%S"),
            _RUN_LOG_FILE.relative_to(_PROJECT_ROOT),
        )

    # ── Console handler ────────────────────────────────────────────────────
    if _LOG_TO_STDOUT:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            _ColorFormatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATE_FMT)
        )
        root.addHandler(console_handler)


_setup_root_logger()

# ── Suppress noisy third-party loggers ───────────────────────────────────────
for _lib in ("httpx", "httpcore", "litellm", "neo4j", "chromadb", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``graphrag`` hierarchy.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` whose output goes to the current run's
        log file (``logs/YYYY-MM-DD/run_HH-MM-SS.log``) and to stdout.

    Example::

        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Ready")
    """
    clean = name.removeprefix("src.").removeprefix("GraphRAGv2.")
    return logging.getLogger(f"graphrag.{clean}")


def current_log_path() -> Path:
    """Return the absolute path of the log file for this run."""
    return _RUN_LOG_FILE

