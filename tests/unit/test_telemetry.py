"""
tests/unit/test_telemetry.py — Unit tests for utils/telemetry.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def telemetry_db(tmp_path, monkeypatch):
    """Redirect telemetry DB to a temp path and return the path."""
    import utils.telemetry as tel_mod

    db_path = tmp_path / "telemetry.db"
    monkeypatch.setattr(tel_mod, "_DB_PATH", db_path)
    tel_mod.init_telemetry_db()
    return db_path


def test_init_creates_table(telemetry_db):
    """init_telemetry_db() must create the events table."""
    conn = sqlite3.connect(telemetry_db)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_track_event_inserts_row(telemetry_db):
    """track_event() must insert exactly one row."""
    from utils.telemetry import track_event

    track_event("test_event", {"key": "val"}, duration_ms=42.5)

    conn = sqlite3.connect(telemetry_db)
    cursor = conn.execute("SELECT event_type, metadata, duration_ms FROM events")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == "test_event"
    assert "key" in rows[0][1]
    assert rows[0][2] == 42.5


def test_track_event_multiple(telemetry_db):
    """Multiple track_event() calls accumulate rows."""
    from utils.telemetry import track_event

    for i in range(5):
        track_event(f"event_{i}")

    conn = sqlite3.connect(telemetry_db)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 5


def test_get_recent_events(telemetry_db):
    """get_recent_events() returns dicts with expected keys."""
    from utils.telemetry import get_recent_events, track_event

    track_event("ranking", {"candidate_count": 3}, duration_ms=123.0)
    events = get_recent_events(limit=5)

    assert len(events) >= 1
    ev = events[0]
    assert "timestamp" in ev
    assert "event_type" in ev
    assert "metadata" in ev
    assert "duration_ms" in ev


def test_track_event_never_raises(telemetry_db, monkeypatch):
    """track_event() must silently swallow any DB error."""
    import utils.telemetry as tel_mod

    # Point to a non-existent path to force DB errors
    monkeypatch.setattr(tel_mod, "_DB_PATH", Path("/nonexistent/path/db.sqlite"))

    # Should not raise
    from utils.telemetry import track_event
    track_event("bad_event", {}, duration_ms=0)
