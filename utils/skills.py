"""
utils/skills.py — Advanced skill extraction engine.

Three-layer matching strategy (applied in order, results merged):
  1. Exact / case-insensitive substring matching   (fast, zero deps)
  2. Fuzzy matching via RapidFuzz                  (handles typos, plurals)
  3. Synonym expansion                              (maps aliases to canonical names)

Optional 4th layer (when SKILL_USE_EMBEDDINGS=true):
  4. Embedding-based similarity                    (catches semantically close terms)

Returns canonical skill names only — no duplicates.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from config import get_logger, settings

logger = get_logger(__name__)


# ── Skill database ────────────────────────────────────────────────────────────


def _load_skill_db() -> list[tuple[str, list[str]]]:
    """
    Load skill definitions from config/skills.json.

    Falls back gracefully if the file is missing or malformed.
    Format: {"skills": [{"name": "Python", "aliases": ["python3"]}]}
    """
    path: Path = settings.skill.db_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        skills = [
            (entry["name"], entry.get("aliases", []))
            for entry in data.get("skills", [])
            if entry.get("name")
        ]
        logger.info("Loaded %d skills from %s", len(skills), path)
        return skills
    except FileNotFoundError:
        logger.warning(
            "skills.json not found at %s — falling back to empty skill DB.", path
        )
        return []
    except Exception as exc:
        logger.warning("Failed to load skills.json (%s) — empty skill DB.", exc)
        return []


# Load at import time
SKILL_DB: list[tuple[str, list[str]]] = _load_skill_db()

# Build lookup maps
_CANONICAL: dict[str, str] = {}          # any form → canonical
_SKILL_SET: set[str] = set()             # all canonical names

for canonical, aliases in SKILL_DB:
    _SKILL_SET.add(canonical)
    _CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _CANONICAL[alias.lower()] = canonical

# Pre-built alias lookup: canonical_name -> list[str aliases]
_ALIASES: dict[str, list[str]] = {canonical: aliases for canonical, aliases in SKILL_DB}


# ── Return type ───────────────────────────────────────────────────────────────

class SkillResult(NamedTuple):
    matched: list[str]   # skills from JD found in resume
    missing: list[str]   # skills from JD not found in resume
    bonus: list[str]     # skills in resume not in JD (nice extras)


# ── Matching layers ───────────────────────────────────────────────────────────


def _exact_match(text: str) -> set[str]:
    """Layer 1 — exact and alias substring matching."""
    text_lower = text.lower()
    found: set[str] = set()

    for form, canonical in _CANONICAL.items():
        # Use word-boundary aware search to avoid "r" matching "reinforcement"
        pattern = r"\b" + re.escape(form) + r"\b"
        if re.search(pattern, text_lower):
            found.add(canonical)

    return found


def _fuzzy_match(text: str, candidates: set[str]) -> set[str]:
    """Layer 2 — fuzzy match of remaining JD skills against resume text tokens."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        logger.debug("rapidfuzz not installed; skipping fuzzy layer.")
        return set()

    # Tokenize resume into n-grams up to length 4
    words = text.split()
    ngrams: list[str] = []
    for n in range(1, 5):
        ngrams.extend(" ".join(words[i: i + n]) for i in range(len(words) - n + 1))

    found: set[str] = set()
    threshold = settings.skill.fuzzy_threshold

    for skill in candidates:
        skill_lower = skill.lower()
        # Also check all aliases
        forms_to_check = [skill_lower] + [
            a.lower() for a in _ALIASES.get(skill, [])
        ]
        for form in forms_to_check:
            result = process.extractOne(
                form, ngrams, scorer=fuzz.token_set_ratio, score_cutoff=threshold
            )
            if result:
                found.add(skill)
                break

    return found


def _embedding_match(skill: str, text: str) -> bool:
    """Layer 3 — embedding cosine similarity for a single skill against resume text."""
    if not settings.skill.use_embeddings:
        return False
    try:
        from utils.embeddings import embed_text
        import numpy as np

        sv = embed_text(skill)
        tv = embed_text(text[:2000])  # limit cost
        sim = float(np.dot(sv, tv) / (np.linalg.norm(sv) * np.linalg.norm(tv) + 1e-9))
        return sim >= settings.skill.embedding_similarity_threshold
    except Exception as exc:
        logger.debug("Embedding skill match failed for '%s': %s", skill, exc)
        return False


# ── Public API ────────────────────────────────────────────────────────────────


def extract_skills(text: str) -> set[str]:
    """Extract all recognizable skills from any text block."""
    return _exact_match(text)


def analyze_skills(resume_text: str, jd_text: str) -> SkillResult:
    """
    Full skill gap analysis.

    Returns:
      matched — JD skills present in resume
      missing — JD skills absent from resume
      bonus   — resume skills not required by JD (differentiators)
    """
    jd_skills = _exact_match(jd_text)
    resume_skills_exact = _exact_match(resume_text)

    # For skills in JD not yet matched, try fuzzy
    unmatched_jd = jd_skills - resume_skills_exact
    resume_fuzzy = _fuzzy_match(resume_text, unmatched_jd)

    resume_skills = resume_skills_exact | resume_fuzzy

    # Optional embedding pass for still-unmatched skills
    still_unmatched = unmatched_jd - resume_fuzzy
    embed_matched: set[str] = set()
    if settings.skill.use_embeddings and still_unmatched:
        for skill in still_unmatched:
            if _embedding_match(skill, resume_text):
                embed_matched.add(skill)
        resume_skills |= embed_matched

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    bonus = sorted(resume_skills - jd_skills)

    logger.debug(
        "Skill analysis: %d JD skills, %d matched, %d missing, %d bonus",
        len(jd_skills), len(matched), len(missing), len(bonus),
    )
    return SkillResult(matched=matched, missing=missing, bonus=bonus)
