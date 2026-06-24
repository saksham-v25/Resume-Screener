"""
tests/unit/test_parser.py — Unit tests for utils/parser.py
"""

from __future__ import annotations

import io

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal real PDF in memory containing the given text."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestParseResume:
    def test_basic_parse_returns_text(self):
        from utils.parser import parse_resume
        pdf_bytes = _make_pdf_bytes("John Doe\nSoftware Engineer\nPython, AWS")
        result = parse_resume(pdf_bytes, "john_doe.pdf")
        assert result["parse_error"] is None
        assert "Python" in result["text"]
        assert result["filename"] == "john_doe.pdf"
        assert result["page_count"] == 1

    def test_name_extraction_title_case(self):
        from utils.parser import parse_resume
        pdf_bytes = _make_pdf_bytes("Alice Johnson\nSenior Data Scientist")
        result = parse_resume(pdf_bytes, "alice.pdf")
        assert result["name"] == "Alice Johnson"

    def test_name_skips_resume_heading(self):
        from utils.parser import parse_resume
        pdf_bytes = _make_pdf_bytes("Resume\nBob Smith\nDevOps Engineer")
        result = parse_resume(pdf_bytes, "bob.pdf")
        assert result["name"] == "Bob Smith"

    def test_corrupt_bytes_returns_error(self):
        from utils.parser import parse_resume
        result = parse_resume(b"this is not a pdf", "bad.pdf")
        assert result["parse_error"] is not None
        assert result["text"] == ""
        assert result["name"] == "Unknown"

    def test_empty_bytes_returns_error(self):
        from utils.parser import parse_resume
        result = parse_resume(b"", "empty.pdf")
        assert result["parse_error"] is not None

    def test_multipage_pdf(self):
        import fitz

        from utils.parser import parse_resume
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 72), f"Page {i + 1} content here.", fontsize=11)
        buf = io.BytesIO()
        doc.save(buf)
        result = parse_resume(buf.getvalue(), "multipage.pdf")
        assert result["page_count"] == 3
        assert result["parse_error"] is None


class TestCleanText:
    def test_collapses_blank_lines(self):
        from utils.parser import _clean_text
        text = "Line 1\n\n\n\n\nLine 2"
        result = _clean_text(text)
        assert "\n\n\n" not in result

    def test_replaces_ligatures(self):
        from utils.parser import _clean_text
        assert _clean_text("ﬁle") == "file"
        assert _clean_text("ﬂow") == "flow"

    def test_fixes_hyphenation(self):
        from utils.parser import _clean_text
        result = _clean_text("devel-\nopment")
        assert "development" in result


class TestExtractName:
    def test_two_word_title_case(self):
        from utils.parser import _extract_name
        assert _extract_name("Jane Smith\nSoftware Engineer") == "Jane Smith"

    def test_four_word_name(self):
        from utils.parser import _extract_name
        assert _extract_name("Mary Ann Von Trapp\nConsultant") == "Mary Ann Von Trapp"

    def test_falls_back_to_unknown(self):
        from utils.parser import _extract_name
        assert _extract_name("123 abc\nno name here") == "Unknown"

    def test_skips_curriculum_vitae(self):
        from utils.parser import _extract_name
        name = _extract_name("Curriculum Vitae\nSam Jones\nEngineer")
        assert name == "Sam Jones"
