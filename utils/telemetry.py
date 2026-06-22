"""
utils/telemetry.py — Lightweight local usage tracking via SQLite.

Events are stored in .cache/telemetry.db.
All writes are fire-and-forget; failures are silently logged — telemetry
must never crash the main application.

Usage:
    from utils.telemetry import init_telemetry_db, track_event

    init_telemetry_db()  # once at startup

    start = time.perf_counter()
    do_something()
    track_event("ranking", {"candidate_count": 5},
                duration_ms=(time.perf_counter() - start) * 1000)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import get_logger

logger = get_logger(__name__)

_DB_PATH: Path | None = None


def _db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        from config import settings
        _DB_PATH = settings.cache.dir / "telemetry.db"
    return _DB_PATH


# ── Init ──────────────────────────────────────────────────────────────────────


def init_telemetry_db() -> None:
    """Create telemetry table if it doesn't exist. Safe to call multiple times."""
    try:
        db = _db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                event_type   TEXT    NOT NULL,
                metadata     TEXT    NOT NULL DEFAULT '{}',
                duration_ms  REAL    NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
        logger.debug("Telemetry DB ready at %s", db)
    except Exception as exc:
        logger.debug("Telemetry DB init failed (non-fatal): %s", exc)


# ── Track ─────────────────────────────────────────────────────────────────────


def track_event(
    event_type: str,
    metadata: dict | None = None,
    duration_ms: float = 0.0,
) -> None:
    """
    Record a telemetry event. Never raises.

    Args:
        event_type:  Short label, e.g. "analysis_run", "ranking", "skill_extract".
        metadata:    Arbitrary JSON-serialisable dict.
        duration_ms: Wall-clock duration in milliseconds.
    """
    try:
        conn = sqlite3.connect(_db_path())
        conn.execute(
            "INSERT INTO events (timestamp, event_type, metadata, duration_ms) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                json.dumps(metadata or {}),
                round(duration_ms, 2),
            ),
        )
        conn.commit()
        conn.close()
        logger.debug("Telemetry: %s (%.0f ms)", event_type, duration_ms)
    except Exception as exc:
        logger.debug("Telemetry track_event failed (non-fatal): %s", exc)


# ── Read (for Status tab) ─────────────────────────────────────────────────────


def get_recent_events(limit: int = 20) -> list[dict]:
    """Return the most recent telemetry events as dicts."""
    try:
        conn = sqlite3.connect(_db_path())
        cursor = conn.execute(
            "SELECT timestamp, event_type, metadata, duration_ms "
            "FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [
            {
                "timestamp": r[0],
                "event_type": r[1],
                "metadata": json.loads(r[2]),
                "duration_ms": r[3],
            }
            for r in cursor.fetchall()
        ]
        conn.close()
        return rows
    except Exception as exc:
        logger.debug("Telemetry get_recent_events failed: %s", exc)
        return []
