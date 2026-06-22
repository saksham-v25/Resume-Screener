"""
tests/unit/test_cache.py — Unit tests for utils/cache.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

DUMMY_RESULT = {
    "name": "Alice",
    "summary": "Excellent candidate.",
    "recommendation": "Recommended",
    "errors": [],
}


def _make_pipeline(result=None):
    """Return a fake pipeline function that records how many times it was called."""
    calls = {"count": 0}

    def pipeline(name, resume_text, jd_text, *args, **kwargs):
        calls["count"] += 1
        return result or DUMMY_RESULT

    pipeline.calls = calls
    return pipeline


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_cache_miss_calls_function(tmp_path, monkeypatch):
    """On first call (cache miss) the wrapped function must be called."""
    import utils.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_result_path", lambda key: tmp_path / f"{key}.pkl")

    fn = _make_pipeline()
    wrapped = cache_mod.cache_agent_result(fn)

    result = wrapped("Alice", "resume text", "jd text")
    assert result == DUMMY_RESULT
    assert fn.calls["count"] == 1


def test_cache_hit_skips_function(tmp_path, monkeypatch):
    """On second call (cache hit) the wrapped function must NOT be called again."""
    import utils.cache as cache_mod
    from config import settings

    monkeypatch.setattr(cache_mod, "_result_path", lambda key: tmp_path / f"{key}.pkl")
    # Make sure cache is enabled
    original = settings.cache.enabled
    object.__setattr__(settings.cache, "enabled", True)

    fn = _make_pipeline()
    wrapped = cache_mod.cache_agent_result(fn)

    # First call — miss
    wrapped("Alice", "resume text", "jd text")
    # Second call — should be a hit
    result = wrapped("Alice", "resume text", "jd text")

    assert result == DUMMY_RESULT
    assert fn.calls["count"] == 1  # only called once

    object.__setattr__(settings.cache, "enabled", original)


def test_cache_disabled_always_calls(tmp_path, monkeypatch):
    """When CACHE_ENABLED=false the function is called every time."""
    import utils.cache as cache_mod
    from config import settings

    monkeypatch.setattr(cache_mod, "_result_path", lambda key: tmp_path / f"{key}.pkl")
    object.__setattr__(settings.cache, "enabled", False)

    fn = _make_pipeline()
    wrapped = cache_mod.cache_agent_result(fn)

    wrapped("Alice", "resume", "jd")
    wrapped("Alice", "resume", "jd")

    assert fn.calls["count"] == 2

    object.__setattr__(settings.cache, "enabled", True)


def test_clear_agent_cache(tmp_path, monkeypatch):
    """clear_agent_cache() should delete all pkl files in the cache dir."""
    import utils.cache as cache_mod

    cache_dir = tmp_path / "agent_results"
    cache_dir.mkdir()
    for i in range(3):
        (cache_dir / f"dummy_{i}.pkl").write_bytes(pickle.dumps({"x": i}))

    monkeypatch.setattr(
        cache_mod, "_result_path", lambda key: cache_dir / f"{key}.pkl"
    )
    # Patch settings.cache.dir so clear_agent_cache finds the right dir
    from config import settings
    monkeypatch.setattr(settings.cache, "dir", tmp_path)

    removed = cache_mod.clear_agent_cache()
    assert removed == 3
    assert list(cache_dir.glob("*.pkl")) == []
