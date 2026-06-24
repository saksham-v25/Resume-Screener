"""
tests/unit/test_database.py — Unit tests for utils/database.py
"""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture()
def analysis_db(tmp_path, monkeypatch):
    """Redirect analysis DB to a temp path and return the path."""
    import utils.database as db_mod

    db_path = tmp_path / "analyses.db"
    monkeypatch.setattr(db_mod, "_DB_PATH", db_path)
    db_mod.init_db()
    return db_path


SAMPLE_CANDIDATES = [
    {
        "name": "Alice",
        "filename": "alice.pdf",
        "score": 85.5,
        "recommendation": "Recommended",
        "matched_skills": ["Python", "Docker"],
        "missing_skills": ["AWS"],
        "summary": "Strong candidate.",
    },
    {
        "name": "Bob",
        "filename": "bob.pdf",
        "score": 60.0,
        "recommendation": "Needs Review",
        "matched_skills": ["Python"],
        "missing_skills": ["Docker", "AWS"],
        "summary": "Mid-level candidate.",
    },
]

SAMPLE_JD = "We need a Python developer with AWS and Docker experience."


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_init_creates_tables(analysis_db):
    """init_db() must create analyses and candidates tables."""
    conn = sqlite3.connect(analysis_db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "analyses" in tables
    assert "candidates" in tables


def test_save_analysis_returns_id(analysis_db):
    """save_analysis() must return a positive integer analysis_id."""
    from utils.database import save_analysis

    aid = save_analysis(SAMPLE_JD, SAMPLE_CANDIDATES)
    assert isinstance(aid, int)
    assert aid > 0


def test_save_analysis_persists_candidates(analysis_db):
    """Saved candidates are retrievable and correct."""
    from utils.database import get_candidates_by_analysis_id, save_analysis

    aid = save_analysis(SAMPLE_JD, SAMPLE_CANDIDATES)
    candidates = get_candidates_by_analysis_id(aid)

    assert len(candidates) == 2
    names = {c["name"] for c in candidates}
    assert "Alice" in names
    assert "Bob" in names


def test_candidates_sorted_by_score_desc(analysis_db):
    """Candidates should be returned highest score first."""
    from utils.database import get_candidates_by_analysis_id, save_analysis

    aid = save_analysis(SAMPLE_JD, SAMPLE_CANDIDATES)
    candidates = get_candidates_by_analysis_id(aid)

    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_get_analysis_history(analysis_db):
    """get_analysis_history() should return saved analyses."""
    from utils.database import get_analysis_history, save_analysis

    save_analysis(SAMPLE_JD, SAMPLE_CANDIDATES)
    save_analysis("Another JD text for second analysis run.", SAMPLE_CANDIDATES[:1])

    history = get_analysis_history(limit=10)
    assert len(history) == 2
    # newest first
    assert history[0]["id"] > history[1]["id"]


def test_matched_skills_deserialized(analysis_db):
    """matched_skills should come back as a Python list, not a JSON string."""
    from utils.database import get_candidates_by_analysis_id, save_analysis

    aid = save_analysis(SAMPLE_JD, SAMPLE_CANDIDATES)
    candidates = get_candidates_by_analysis_id(aid)

    alice = next(c for c in candidates if c["name"] == "Alice")
    assert isinstance(alice["matched_skills"], list)
    assert "Python" in alice["matched_skills"]


def test_save_analysis_handles_empty_candidates(analysis_db):
    """save_analysis() should not crash with an empty candidate list."""
    from utils.database import save_analysis

    aid = save_analysis(SAMPLE_JD, [])
    assert aid is not None
