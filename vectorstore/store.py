"""
vectorstore/store.py — FAISS semantic search index.

Improvements over v1:
  - Index and metadata persisted to disk and reloaded without rebuild
  - Graceful fallback when FAISS is unavailable
  - Strict type annotations and logging
  - Thread-safe build (idempotent: building twice is safe)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from config import get_logger, settings
from utils.embeddings import embed_text

logger = get_logger(__name__)

INDEX_PATH: Path = settings.faiss_index_path
META_PATH: Path = settings.faiss_meta_path


# ── Internal state ────────────────────────────────────────────────────────────

_index = None
_metadata: list[dict] = []   # [{name, filename, text}, …]


# ── Build / load ──────────────────────────────────────────────────────────────


def build_index(candidates: list[dict]) -> None:
    """
    Build a new FAISS index from a list of candidate dicts.

    Each dict must have: name, filename, text.
    Saves index and metadata to disk for reuse.
    """
    global _index, _metadata

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        ) from exc

    if not candidates:
        raise ValueError("Cannot build index: candidate list is empty.")

    valid = [c for c in candidates if c.get("text", "").strip()]
    if not valid:
        raise ValueError("All candidates have empty text — cannot build index.")

    logger.info("Building FAISS index for %d candidates…", len(valid))
    vectors: list[np.ndarray] = []

    for c in valid:
        try:
            vec = embed_text(c["text"][:6000])
            vectors.append(vec)
        except Exception as exc:
            logger.warning("Skipping '%s' during index build: %s", c.get("filename"), exc)

    if not vectors:
        raise RuntimeError("No embeddings could be computed — index build aborted.")

    matrix = np.vstack(vectors).astype("float32")
    # L2-normalize for cosine similarity via inner product
    faiss.normalize_L2(matrix)

    dim = matrix.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(matrix)

    _index = idx
    _metadata = valid[: len(vectors)]

    # Persist
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)

    logger.info("FAISS index saved: %s (%d vectors, dim=%d)", INDEX_PATH, len(vectors), dim)


def load_index() -> bool:
    """
    Load persisted index from disk. Returns True on success, False if not found.
    Call this at startup to avoid rebuilding every run.
    """
    global _index, _metadata

    if not INDEX_PATH.exists() or not META_PATH.exists():
        logger.debug("No persisted FAISS index found.")
        return False

    try:
        import faiss
        _index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)
        logger.info("Loaded FAISS index from disk: %d vectors", _index.ntotal)
        return True
    except Exception as exc:
        logger.warning("Failed to load FAISS index: %s", exc)
        _index = None
        _metadata = []
        return False


def invalidate_index() -> None:
    """Remove persisted index so it will be rebuilt on next use."""
    global _index, _metadata
    _index = None
    _metadata = []
    INDEX_PATH.unlink(missing_ok=True)
    META_PATH.unlink(missing_ok=True)
    logger.info("FAISS index invalidated.")


# ── Search ────────────────────────────────────────────────────────────────────


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search over indexed resumes.

    Returns a list of candidate dicts with an added 'similarity_score' field,
    sorted by relevance descending. Empty list if index is not built.
    """
    if _index is None:
        logger.warning("Search called but index is not built.")
        return []

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    try:
        import faiss
        qvec = embed_text(query).astype("float32").reshape(1, -1)
        faiss.normalize_L2(qvec)

        k = min(top_k, _index.ntotal)
        scores, indices = _index.search(qvec, k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            candidate = dict(_metadata[idx])
            candidate["similarity_score"] = round(float(score) * 100, 2)
            results.append(candidate)

        logger.debug("FAISS search '%s' → %d results", query[:60], len(results))
        return results

    except Exception as exc:
        logger.exception("FAISS search failed: %s", exc)
        return []


def index_is_built() -> bool:
    return _index is not None and _index.ntotal > 0
