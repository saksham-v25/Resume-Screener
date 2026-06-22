"""
utils/database.py — SQLite persistence layer for analysis results.

Schema:
  analyses  — one row per analysis run (JD + timestamp)
  candidates — one row per candidate per analysis

Usage:
    from utils.database import init_db, save_analysis, get_analysis_history

    init_db()  # once at startup
    analysis_id = save_analysis(jd_text, candidates)
    history = get_analysis_history(limit=10)
"""

from __future__ import annotations

import hashlib
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
        _DB_PATH = settings.cache.dir / "analyses.db"
    return _DB_PATH


# ── Init ──────────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    try:
        db = _db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                jd_hash         TEXT    NOT NULL,
                jd_snippet      TEXT    NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id   INTEGER NOT NULL,
                name          TEXT    NOT NULL,
                filename      TEXT    NOT NULL DEFAULT '',
                score         REAL    NOT NULL DEFAULT 0,
                recommendation TEXT   NOT NULL DEFAULT 'Needs Review',
                matched_skills TEXT   NOT NULL DEFAULT '[]',
                missing_skills TEXT   NOT NULL DEFAULT '[]',
                summary        TEXT   NOT NULL DEFAULT '',
                created_at     TEXT   NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            )
        """)
        conn.commit()
        conn.close()
        logger.debug("Analysis DB ready at %s", db)
    except Exception as exc:
        logger.warning("Analysis DB init failed (non-fatal): %s", exc)


# ── Write ─────────────────────────────────────────────────────────────────────


def save_analysis(jd_text: str, candidates: list[dict]) -> int | None:
    """
    Persist an analysis run.

    Returns the new analysis_id, or None on failure.
    """
    try:
        conn = sqlite3.connect(_db_path())
        jd_hash = hashlib.sha256(jd_text.encode()).hexdigest()
        jd_snippet = jd_text[:200].replace("\n", " ").strip()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            "INSERT INTO analyses (created_at, jd_hash, jd_snippet, candidate_count) "
            "VALUES (?, ?, ?, ?)",
            (now, jd_hash, jd_snippet, len(candidates)),
        )
        analysis_id = cursor.lastrowid

        for c in candidates:
            conn.execute(
                "INSERT INTO candidates "
                "(analysis_id, name, filename, score, recommendation, "
                " matched_skills, missing_skills, summary, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis_id,
                    c.get("name", "Unknown"),
                    c.get("filename", ""),
                    round(float(c.get("score", 0)), 2),
                    c.get("recommendation", "Needs Review"),
                    json.dumps(c.get("matched_skills", [])),
                    json.dumps(c.get("missing_skills", [])),
                    c.get("summary", ""),
                    now,
                ),
            )

        conn.commit()
        conn.close()
        logger.info("Saved analysis #%d (%d candidates)", analysis_id, len(candidates))
        return analysis_id
    except Exception as exc:
        logger.warning("save_analysis failed (non-fatal): %s", exc)
        return None


# ── Read ──────────────────────────────────────────────────────────────────────


def get_analysis_history(limit: int = 20) -> list[dict]:
    """Return the most recent analysis runs, newest first."""
    try:
        conn = sqlite3.connect(_db_path())
        cursor = conn.execute(
            "SELECT id, created_at, jd_snippet, candidate_count "
            "FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [
            {
                "id": r[0],
                "created_at": r[1],
                "jd_snippet": r[2],
                "candidate_count": r[3],
            }
            for r in cursor.fetchall()
        ]
        conn.close()
        return rows
    except Exception as exc:
        logger.debug("get_analysis_history failed: %s", exc)
        return []


def get_candidates_by_analysis_id(analysis_id: int) -> list[dict]:
    """Return all candidates for a given analysis, ranked by score desc."""
    try:
        conn = sqlite3.connect(_db_path())
        cursor = conn.execute(
            "SELECT name, filename, score, recommendation, "
            "       matched_skills, missing_skills, summary "
            "FROM candidates WHERE analysis_id = ? ORDER BY score DESC",
            (analysis_id,),
        )
        rows = [
            {
                "name": r[0],
                "filename": r[1],
                "score": r[2],
                "recommendation": r[3],
                "matched_skills": json.loads(r[4]),
                "missing_skills": json.loads(r[5]),
                "summary": r[6],
            }
            for r in cursor.fetchall()
        ]
        conn.close()
        return rows
    except Exception as exc:
        logger.debug("get_candidates_by_analysis_id failed: %s", exc)
        return []
