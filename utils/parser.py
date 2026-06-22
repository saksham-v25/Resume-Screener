"""
utils/parser.py — PDF parsing with text preprocessing and robust error handling.

Improvements over v1:
  - Strips headers/footers heuristically (repeated lines across pages)
  - Cleans common PDF artefacts (hyphenation, ligatures, excessive whitespace)
  - Graceful handling of corrupt/password-protected PDFs
  - Structured return type (TypedDict)
  - Full logging
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF

from config import get_logger

logger = get_logger(__name__)

# ── Return type ──────────────────────────────────────────────────────────────


class ParsedResume(TypedDict):
    name: str
    filename: str
    text: str
    page_count: int
    parse_error: str | None


# ── Public API ───────────────────────────────────────────────────────────────


def parse_resume(file_bytes: bytes, filename: str) -> ParsedResume:
    """
    Parse a PDF resume from raw bytes.

    Returns a ParsedResume dict. On failure, text is empty and
    parse_error contains a human-readable message — never raises.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("Failed to open PDF '%s': %s", filename, exc)
        return _error_result(filename, f"Could not open PDF: {exc}")

    if doc.is_encrypted:
        logger.warning("PDF '%s' is password-protected — skipping.", filename)
        return _error_result(filename, "PDF is password-protected.")

    try:
        raw_pages = _extract_pages(doc)
        cleaned_pages = _remove_repeated_lines(raw_pages)
        full_text = _clean_text("\n".join(cleaned_pages))
        candidate_name = _extract_name(full_text)
        page_count = len(doc)
    except Exception as exc:
        logger.exception("Unexpected error parsing '%s'", filename)
        return _error_result(filename, f"Parse error: {exc}")
    finally:
        doc.close()

    logger.info("Parsed '%s' → name='%s', pages=%d, chars=%d",
                filename, candidate_name, page_count, len(full_text))

    return ParsedResume(
        name=candidate_name,
        filename=filename,
        text=full_text,
        page_count=page_count,
        parse_error=None,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _extract_pages(doc: fitz.Document) -> list[str]:
    """Extract raw text from every page."""
    pages = []
    for page in doc:
        try:
            pages.append(page.get_text("text"))
        except Exception as exc:
            logger.debug("Skipping page %d: %s", page.number, exc)
            pages.append("")
    return pages


def _remove_repeated_lines(pages: list[str]) -> list[str]:
    """
    Heuristically strip headers/footers: lines that appear in ≥60 % of pages
    and are short (≤ 80 chars) are treated as boilerplate.
    """
    if len(pages) < 3:
        return pages

    line_freq: Counter[str] = Counter()
    for page_text in pages:
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) <= 80:
                line_freq[stripped] += 1

    threshold = max(2, int(len(pages) * 0.6))
    boilerplate = {line for line, count in line_freq.items() if count >= threshold}

    cleaned = []
    for page_text in pages:
        lines = [
            ln for ln in page_text.splitlines()
            if ln.strip() not in boilerplate
        ]
        cleaned.append("\n".join(lines))
    return cleaned


def _clean_text(text: str) -> str:
    """Normalize whitespace, fix hyphenation, replace common ligatures."""
    # Common PDF ligatures → ASCII
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)

    # De-hyphenate line-break hyphenation (e.g. "devel-\nopment" → "development")
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove non-printable characters except newlines/tabs
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    return text.strip()


def _extract_name(text: str) -> str:
    """
    Heuristic name extraction from the first non-empty lines.
    Looks for a line that looks like "FirstName LastName" (2–4 words, title case).
    Falls back to 'Unknown'.
    """
    for line in text.splitlines()[:10]:
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            # Skip lines that look like headings ("Curriculum Vitae", "Resume", etc.)
            skip = {"resume", "curriculum", "vitae", "cv", "profile", "summary"}
            if not any(w.lower() in skip for w in words):
                return line
    return "Unknown"


def _error_result(filename: str, message: str) -> ParsedResume:
    return ParsedResume(
        name="Unknown",
        filename=filename,
        text="",
        page_count=0,
        parse_error=message,
    )
