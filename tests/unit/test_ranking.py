"""
tests/unit/test_ranking.py — Unit tests for utils/ranking.py
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _make_resume(name: str, filename: str, text: str, error: str | None = None):
    return {
        "name": name,
        "filename": filename,
        "text": text,
        "page_count": 1,
        "parse_error": error,
    }


def _mock_embed(text: str) -> np.ndarray:
    """Deterministic mock: hash text into a fixed vector."""
    rng = np.random.default_rng(abs(hash(text[:50])) % (2**31))
    vec = rng.random(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


class TestRankCandidates:
    def test_returns_sorted_descending(self):
        from utils.ranking import rank_candidates
        resumes = [
            _make_resume("Alice", "alice.pdf", "Python AWS Docker ML engineer"),
            _make_resume("Bob", "bob.pdf", "Java Spring Boot enterprise"),
            _make_resume("Carol", "carol.pdf", "Python machine learning TensorFlow"),
        ]
        jd = "Looking for a Python machine learning engineer with AWS experience."

        with patch("utils.ranking.embed_text", side_effect=_mock_embed):
            results = rank_candidates(resumes, jd)

        assert len(results) == 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_is_percentage(self):
        from utils.ranking import rank_candidates
        resumes = [_make_resume("Alice", "alice.pdf", "Python developer")]
        jd = "Python developer needed"

        with patch("utils.ranking.embed_text", side_effect=_mock_embed):
            results = rank_candidates(resumes, jd)

        assert 0.0 <= results[0]["score"] <= 100.0

    def test_skips_parse_error_resumes(self):
        from utils.ranking import rank_candidates
        resumes = [
            _make_resume("Alice", "alice.pdf", "Python developer"),
            _make_resume("Bad", "bad.pdf", "", error="Corrupt PDF"),
        ]
        jd = "Python developer"

        with patch("utils.ranking.embed_text", side_effect=_mock_embed):
            results = rank_candidates(resumes, jd)

        filenames = [r["filename"] for r in results]
        assert "bad.pdf" not in filenames

    def test_empty_jd_raises(self):
        from utils.ranking import rank_candidates
        resumes = [_make_resume("Alice", "alice.pdf", "Python")]
        with pytest.raises(ValueError, match="Job description"):
            rank_candidates(resumes, "")

    def test_all_bad_resumes_raises(self):
        from utils.ranking import rank_candidates
        resumes = [_make_resume("Bad", "bad.pdf", "", error="Parse failed")]
        with pytest.raises(ValueError, match="No valid resumes"):
            rank_candidates(resumes, "Python developer")

    def test_embed_failure_gives_zero_score(self):
        from utils.ranking import rank_candidates
        resumes = [
            _make_resume("Alice", "alice.pdf", "Python"),
            _make_resume("Bob", "bob.pdf", "Java"),
        ]
        jd = "Python developer"

        call_count = 0

        def mock_embed(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:          # JD embed succeeds
                return _mock_embed(text)
            if "Java" in text:            # Bob's embed fails
                raise RuntimeError("Embedding service down")
            return _mock_embed(text)

        with patch("utils.ranking.embed_text", side_effect=mock_embed):
            results = rank_candidates(resumes, jd)

        bob = next(r for r in results if r["filename"] == "bob.pdf")
        assert bob["score"] == 0.0
        assert bob["embed_error"] is not None
