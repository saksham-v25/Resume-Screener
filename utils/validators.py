"""
utils/validators.py — Input validation for JD and resume text.

Uses Pydantic v2 models to enforce size limits and basic content sanity checks.
Raises pydantic.ValidationError on invalid input — callers should catch and
display user-friendly messages.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field, field_validator

# ── Models ────────────────────────────────────────────────────────────────────


class JobDescriptionInput(BaseModel):
    """Validated job description text."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="Job description text (10 – 50 000 chars)",
    )

    @field_validator("text")
    @classmethod
    def not_spam(cls, v: str) -> str:
        """Reject if more than 50 % of non-empty lines are identical (spam guard)."""
        lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
        if len(lines) > 5:
            most_common_count = Counter(lines).most_common(1)[0][1]
            if most_common_count / len(lines) > 0.5:
                raise ValueError(
                    "Job description appears to be repetitive or spam. "
                    "Please paste a real job description."
                )
        return v.strip()


class ResumeInput(BaseModel):
    """Validated parsed resume text."""

    filename: str = Field(..., max_length=255)
    text: str = Field(
        ...,
        min_length=20,
        max_length=100_000,
        description="Parsed resume text (20 – 100 000 chars)",
    )


# ── Convenience helpers ───────────────────────────────────────────────────────


def validate_jd(text: str) -> tuple[bool, str]:
    """
    Validate JD text. Returns (ok, error_message).
    error_message is empty string when ok=True.
    """
    try:
        JobDescriptionInput(text=text)
        return True, ""
    except Exception as exc:
        # Extract first validation message
        msgs = []
        if hasattr(exc, "errors"):
            msgs = [e.get("msg", str(e)) for e in exc.errors()]
        return False, msgs[0] if msgs else str(exc)


def validate_resume(filename: str, text: str) -> tuple[bool, str]:
    """Validate resume text. Returns (ok, error_message)."""
    try:
        ResumeInput(filename=filename, text=text)
        return True, ""
    except Exception as exc:
        msgs = []
        if hasattr(exc, "errors"):
            msgs = [e.get("msg", str(e)) for e in exc.errors()]
        return False, msgs[0] if msgs else str(exc)
