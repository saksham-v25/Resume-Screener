"""
tests/unit/test_validators.py — Unit tests for utils/validators.py
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from utils.validators import JobDescriptionInput, ResumeInput, validate_jd, validate_resume


# ── JobDescriptionInput ───────────────────────────────────────────────────────


def test_valid_jd_passes():
    jd = JobDescriptionInput(text="We need a Python developer with 5+ years experience in AWS and Docker.")
    assert "Python" in jd.text


def test_jd_too_short_rejected():
    with pytest.raises(ValidationError):
        JobDescriptionInput(text="Hi")


def test_jd_empty_rejected():
    with pytest.raises(ValidationError):
        JobDescriptionInput(text="")


def test_jd_too_long_rejected():
    with pytest.raises(ValidationError):
        JobDescriptionInput(text="x" * 50_001)


def test_jd_spam_rejected():
    spam_line = "Apply now for this great job!\n"
    spam_text = spam_line * 20
    with pytest.raises(ValidationError, match="repetitive"):
        JobDescriptionInput(text=spam_text)


def test_jd_strips_whitespace():
    jd = JobDescriptionInput(text="  Python developer needed.  ")
    assert jd.text == "Python developer needed."


# ── ResumeInput ───────────────────────────────────────────────────────────────


def test_valid_resume_passes():
    r = ResumeInput(filename="alice.pdf", text="Alice Smith. Python developer with 5 years experience.")
    assert r.filename == "alice.pdf"


def test_resume_too_short_rejected():
    with pytest.raises(ValidationError):
        ResumeInput(filename="a.pdf", text="Short")


def test_resume_filename_too_long_rejected():
    with pytest.raises(ValidationError):
        ResumeInput(filename="a" * 256, text="Alice Smith. Python developer with 5 years experience.")


# ── validate_jd / validate_resume helpers ─────────────────────────────────────


def test_validate_jd_ok():
    ok, err = validate_jd("We need a Python developer with strong AWS experience and Docker knowledge.")
    assert ok is True
    assert err == ""


def test_validate_jd_fail():
    ok, err = validate_jd("x")
    assert ok is False
    assert err != ""


def test_validate_resume_ok():
    ok, err = validate_resume("cv.pdf", "Jane Doe. Senior Python engineer with 6 years of experience.")
    assert ok is True


def test_validate_resume_fail():
    ok, err = validate_resume("cv.pdf", "Hi")
    assert ok is False
    assert err != ""
