"""
utils/cache.py — LLM agent result caching.

Caches the output of run_agent_pipeline() keyed by SHA-256(name + resume + jd).
Results are stored as pickle files in .cache/agent_results/.

Enabled/disabled via CACHE_ENABLED env var (default: true).
"""

from __future__ import annotations

import hashlib
import pickle
from functools import wraps
from pathlib import Path

from config import get_logger, settings

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _result_key(name: str, resume_text: str, jd_text: str) -> str:
    """Stable SHA-256 hash for a (name, resume, jd) triple."""
    payload = f"{name}:{resume_text}:{jd_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result_path(key: str) -> Path:
    return settings.cache.dir / "agent_results" / f"{key}.pkl"


# ── Decorator ─────────────────────────────────────────────────────────────────


def cache_agent_result(func):
    """
    Decorator that caches run_agent_pipeline() results to disk.

    - Cache hit  → return stored result immediately (no LLM calls).
    - Cache miss → call the real function, persist result, return it.
    - Always passes through when CACHE_ENABLED=false.
    """

    @wraps(func)
    def wrapper(
        name: str,
        resume_text: str,
        jd_text: str,
        *args,
        **kwargs,
    ):
        if not settings.cache.enabled:
            return func(name, resume_text, jd_text, *args, **kwargs)

        key = _result_key(name, resume_text, jd_text)
        cache_path = _result_path(key)

        # ── Try load from cache ─────────────────────────────────────────────
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                logger.info("Agent result cache HIT for '%s' (key=%s…)", name, key[:12])
                return result
            except Exception as exc:
                logger.debug("Cache load failed for key %s: %s", key[:12], exc)
                # Fall through to re-compute

        # ── Compute ─────────────────────────────────────────────────────────
        result = func(name, resume_text, jd_text, *args, **kwargs)

        # ── Persist ─────────────────────────────────────────────────────────
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            logger.debug("Agent result cached for '%s' (key=%s…)", name, key[:12])
        except Exception as exc:
            logger.debug("Cache save failed: %s", exc)

        return result

    return wrapper


def clear_agent_cache() -> int:
    """Delete all cached agent results. Returns number of files removed."""
    cache_dir = settings.cache.dir / "agent_results"
    if not cache_dir.exists():
        return 0
    count = 0
    for f in cache_dir.glob("*.pkl"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    logger.info("Cleared %d cached agent results.", count)
    return count
