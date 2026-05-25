"""Centralized logging setup for GraphRAGv2."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_LOG_ROOT      = _PROJECT_ROOT / "logs"

_LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_TO_FILE   = os.getenv("LOG_TO_FILE",   "true").lower() != "false"
_LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "true").lower() != "false"

# Timestamp captured once at process start → shared by all loggers in this run
_RUN_START     = datetime.now()
_DATE_FOLDER   = _RUN_START.strftime("%Y-%m-%d")
_RUN_FILENAME  = _RUN_START.strftime("run_%H-%M-%S.log")
_RUN_LOG_FILE  = _LOG_ROOT / _DATE_FOLDER / _RUN_FILENAME

_FILE_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATE_FMT  = "%H:%M:%S"

_CONSOLE_FORMAT   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_DATE_FMT = "%H:%M:%S"

_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Work on a copy so we don't mutate the shared LogRecord
        record = logging.makeLogRecord(record.__dict__)
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def _setup_root_logger() -> None:
    root = logging.getLogger("graphrag")

    if root.handlers:           # already configured (e.g. Streamlit hot-reload)
        return

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    root.propagate = False

    if _LOG_TO_FILE:
        _RUN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            filename=str(_RUN_LOG_FILE),
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FMT)
        )
        root.addHandler(file_handler)

        root.info(
            "══ Run started at %s ══  log → %s",
            _RUN_START.strftime("%Y-%m-%d %H:%M:%S"),
            _RUN_LOG_FILE.relative_to(_PROJECT_ROOT),
        )

    if _LOG_TO_STDOUT:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            _ColorFormatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATE_FMT)
        )
        root.addHandler(console_handler)


_setup_root_logger()

for _lib in ("httpx", "httpcore", "litellm", "neo4j", "chromadb", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``graphrag`` hierarchy."""
    clean = name.removeprefix("src.").removeprefix("GraphRAGv2.")
    return logging.getLogger(f"graphrag.{clean}")


def current_log_path() -> Path:
    """Return the absolute path of the log file for this run."""
    return _RUN_LOG_FILE
