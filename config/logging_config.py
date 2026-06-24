"""
Structured logging configuration.
Call setup_logging() once at app startup; then use get_logger(__name__) everywhere.

Set LOG_FORMAT=json in .env for production JSON log output (Splunk/ELK compatible).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── Formatters ────────────────────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line — ideal for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


_TEXT_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Setup ─────────────────────────────────────────────────────────────────────


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    json_output: bool = False,
) -> None:
    """
    Configure root logger.

    Args:
        log_level:   DEBUG / INFO / WARNING / ERROR
        log_dir:     If provided, also write to <log_dir>/app.log
        json_output: Emit JSON lines instead of human-readable text
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter: logging.Formatter = JSONFormatter() if json_output else _TEXT_FMT

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        fh.setLevel(level)
        handlers.append(fh)

    for h in handlers:
        h.setFormatter(formatter)
        h.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
