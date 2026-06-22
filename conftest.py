"""
conftest.py — Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for all tests
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Environment defaults for tests ────────────────────────────────────────────

os.environ.setdefault("EMBEDDING_BACKEND", "huggingface")
os.environ.setdefault("LLM_BACKEND", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-key-placeholder")
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("SKILL_USE_EMBEDDINGS", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def sample_jd() -> str:
    return """
    Senior Python Developer required.
    Skills: Python, Docker, AWS, FastAPI, PostgreSQL, Machine Learning.
    5+ years experience. Team player. Strong problem-solving.
    """


@pytest.fixture()
def sample_resume_text() -> str:
    return """
    Jane Doe
    Senior Software Engineer

    Skills: Python, Docker, FastAPI, PostgreSQL, scikit-learn, Kubernetes
    Experience: 6 years building ML-powered APIs and data pipelines.
    AWS deployments, CI/CD with GitHub Actions.
    """


@pytest.fixture()
def sample_parsed_resume(sample_resume_text) -> dict:
    return {
        "name": "Jane Doe",
        "filename": "jane_doe.pdf",
        "text": sample_resume_text,
        "page_count": 1,
        "parse_error": None,
    }


@pytest.fixture()
def minimal_pdf_bytes() -> bytes:
    """Returns bytes of a real minimal single-page PDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Alice Smith\nPython Developer\nAWS Docker ML", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
