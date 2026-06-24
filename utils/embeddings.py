"""
utils/embeddings.py — Embedding engine with dual backend support and disk caching.

Backends:
  - ollama   : nomic-embed-text via local Ollama server
  - huggingface : sentence-transformers model (no external server needed)

Caching:
  - Embeddings are SHA-256 keyed and saved to .cache/ to avoid recomputation.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Literal

import numpy as np

from config import get_logger, settings

logger = get_logger(__name__)

EmbeddingBackend = Literal["ollama", "huggingface"]

# ── Cache helpers ─────────────────────────────────────────────────────────────


def _cache_key(text: str, backend: str, model: str) -> str:
    payload = f"{backend}:{model}:{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return settings.cache.dir / f"{key}.npy"


def _load_from_cache(key: str) -> np.ndarray | None:
    if not settings.cache.enabled:
        return None
    p = _cache_path(key)
    if p.exists():
        try:
            vec = np.load(str(p))
            logger.debug("Cache hit: %s", key[:12])
            return vec
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _save_to_cache(key: str, vec: np.ndarray) -> None:
    if not settings.cache.enabled:
        return
    try:
        np.save(str(_cache_path(key)), vec)
    except Exception as exc:
        logger.debug("Cache write failed: %s", exc)


# ── Ollama backend ────────────────────────────────────────────────────────────


def _embed_ollama(text: str) -> np.ndarray:
    import requests  # lazily imported

    url = f"{settings.embedding.ollama_base_url}/api/embeddings"
    payload = {"model": settings.embedding.ollama_model, "prompt": text}

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype=np.float32)
        return vec
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {settings.embedding.ollama_base_url}. "
            "Start Ollama or switch EMBEDDING_BACKEND=huggingface."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama embedding failed: {exc}") from exc


def check_ollama_health() -> tuple[bool, str]:
    """Returns (is_healthy, message)."""
    import requests

    try:
        resp = requests.get(settings.embedding.ollama_base_url, timeout=5)
        if resp.status_code == 200:
            return True, "Ollama is running."
        return False, f"Ollama returned HTTP {resp.status_code}."
    except Exception as exc:
        return False, f"Ollama unreachable: {exc}"


# ── HuggingFace backend ───────────────────────────────────────────────────────

_hf_model_cache: dict[str, object] = {}


def _get_hf_model():
    model_name = settings.embedding.hf_model
    if model_name not in _hf_model_cache:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading HuggingFace model: %s", model_name)
            _hf_model_cache[model_name] = SentenceTransformer(model_name)
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
    return _hf_model_cache[model_name]


def _embed_huggingface(text: str) -> np.ndarray:
    model = _get_hf_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single string. Returns a float32 numpy vector.
    Uses disk cache to avoid recomputation.
    Raises RuntimeError with a user-friendly message on backend failure.
    """
    backend = settings.embedding.backend
    model = (
        settings.embedding.ollama_model
        if backend == "ollama"
        else settings.embedding.hf_model
    )
    key = _cache_key(text, backend, model)
    cached = _load_from_cache(key)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    if backend == "ollama":
        vec = _embed_ollama(text)
    elif backend == "huggingface":
        vec = _embed_huggingface(text)
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}")

    elapsed = time.perf_counter() - t0
    logger.debug("Embedded %d chars via %s in %.2fs", len(text), backend, elapsed)
    _save_to_cache(key, vec)
    return vec


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Batch embed (sequential). Returns list of vectors in the same order."""
    return [embed_text(t) for t in texts]


def embed_texts_batch(
    texts: list[str],
    max_workers: int = 4,
) -> list[np.ndarray]:
    """
    Parallel batch embedding using a thread-pool.

    Falls back to sequential embed_texts() if concurrency fails.
    Preserves input order.

    Args:
        texts:       List of strings to embed.
        max_workers: Thread pool size (default 4; HuggingFace model is thread-safe).

    Returns:
        List of float32 numpy vectors, same length and order as `texts`.
    """
    if not texts:
        return []

    if len(texts) == 1:
        return [embed_text(texts[0])]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[np.ndarray | None] = [None] * len(texts)
    workers = min(max_workers, len(texts))

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(embed_text, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Batch embed failed for text[%d]: %s — retrying sequentially.", idx, exc
                    )
                    results[idx] = embed_text(texts[idx])

        logger.debug(
            "Batch embedded %d texts with %d workers.", len(texts), workers
        )
        return [r for r in results if r is not None]

    except Exception as exc:
        logger.warning(
            "embed_texts_batch crashed (%s) — falling back to sequential.", exc
        )
        return embed_texts(texts)
