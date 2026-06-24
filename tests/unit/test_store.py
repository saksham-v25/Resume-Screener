"""
tests/unit/test_store.py — Unit tests for vectorstore/store.py
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _mock_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text[:50])) % (2**31))
    vec = rng.random(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _sample_candidates():
    return [
        {"name": "Alice", "filename": "alice.pdf", "text": "Python ML engineer AWS"},
        {"name": "Bob", "filename": "bob.pdf", "text": "Java Spring Boot developer"},
        {"name": "Carol", "filename": "carol.pdf", "text": "Data scientist TensorFlow NLP"},
    ]


class TestBuildIndex:
    def test_builds_without_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CACHE_ENABLED", "false")
        from vectorstore import store
        monkeypatch.setattr(store, "INDEX_PATH", tmp_path / "index.faiss")
        monkeypatch.setattr(store, "META_PATH", tmp_path / "meta.pkl")
        store._index = None
        store._metadata = []

        with patch("vectorstore.store.embed_text", side_effect=_mock_embed):
            store.build_index(_sample_candidates())

        assert store.index_is_built()
        assert store._index.ntotal == 3

    def test_raises_on_empty_candidates(self):
        from vectorstore import store
        with pytest.raises(ValueError, match="empty"):
            store.build_index([])

    def test_raises_on_all_empty_text(self):
        from vectorstore import store
        bad = [{"name": "X", "filename": "x.pdf", "text": ""}]
        with pytest.raises(ValueError, match="empty text"):
            store.build_index(bad)


class TestSearch:
    def setup_method(self):
        """Build a fresh in-memory index before each test."""
        from vectorstore import store
        with patch("vectorstore.store.embed_text", side_effect=_mock_embed):
            # Directly populate without disk I/O for speed
            import faiss
            candidates = _sample_candidates()
            vecs = [_mock_embed(c["text"]) for c in candidates]
            matrix = np.vstack(vecs).astype("float32")
            faiss.normalize_L2(matrix)
            idx = faiss.IndexFlatIP(768)
            idx.add(matrix)
            store._index = idx
            store._metadata = candidates

    def test_returns_top_k_results(self):
        from vectorstore import store
        with patch("vectorstore.store.embed_text", side_effect=_mock_embed):
            results = store.search("machine learning Python", top_k=2)
        assert len(results) == 2

    def test_results_have_similarity_score(self):
        from vectorstore import store
        with patch("vectorstore.store.embed_text", side_effect=_mock_embed):
            results = store.search("Python ML", top_k=3)
        for r in results:
            assert "similarity_score" in r
            assert 0.0 <= r["similarity_score"] <= 100.0

    def test_empty_query_raises(self):
        from vectorstore import store
        with pytest.raises(ValueError, match="empty"):
            store.search("")

    def test_empty_index_returns_empty(self):
        from vectorstore import store
        store._index = None
        results = store.search("Python developer")
        assert results == []
