"""
utils/report.py — PDF and CSV report generation for recruitment results.

Generates a structured PDF report using fpdf2 with:
  - Header with title, date and JD snippet
  - Per-candidate sections: score, recommendation, matched/missing skills, summary

Falls back gracefully if fpdf2 is not installed.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from config import get_logger

logger = get_logger(__name__)


# ── PDF Report ────────────────────────────────────────────────────────────────


def generate_pdf(candidates: list[dict], jd_text: str) -> bytes:
    """
    Generate a PDF recruitment report.

    Args:
        candidates: List of candidate dicts (from analysis pipeline).
        jd_text:    Raw job description text.

    Returns:
        PDF content as bytes.

    Raises:
        RuntimeError: If fpdf2 is not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError(
            "fpdf2 is not installed. Run: pip install fpdf2"
        ) from exc

    class _PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "AI Recruiter Assistant — Recruitment Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(100, 100, 100)
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            self.cell(0, 6, f"Generated: {date_str}", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")
            self.set_text_color(0, 0, 0)

    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── JD Snippet ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Job Description (excerpt)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(240, 240, 240)
    snippet = jd_text[:500].replace("\r", "").strip()
    if len(jd_text) > 500:
        snippet += "…"
    pdf.multi_cell(0, 5, snippet, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Candidate Table Header ────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Candidate Rankings ({len(candidates)} total)", new_x="LMARGIN", new_y="NEXT")

    _BADGE = {
        "Recommended":     "✓ Recommended",
        "Not Recommended": "✗ Not Recommended",
        "Needs Review":    "? Needs Review",
    }

    for rank, c in enumerate(candidates, 1):
        pdf.ln(3)
        pdf.set_fill_color(230, 235, 245)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(
            0, 8,
            f"#{rank}  {c.get('name', 'Unknown')}   |   Score: {c.get('score', 0):.1f}%   |   "
            f"{_BADGE.get(c.get('recommendation', ''), c.get('recommendation', ''))}",
            fill=True,
            new_x="LMARGIN", new_y="NEXT",
        )

        pdf.set_font("Helvetica", "", 9)

        # Matched skills
        matched = ", ".join(c.get("matched_skills", [])) or "None"
        pdf.set_text_color(0, 100, 0)
        pdf.multi_cell(0, 5, f"  Matched Skills: {matched}", new_x="LMARGIN", new_y="NEXT")

        # Missing skills
        missing = ", ".join(c.get("missing_skills", [])) or "None"
        pdf.set_text_color(180, 0, 0)
        pdf.multi_cell(0, 5, f"  Missing Skills: {missing}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        # Summary
        summary = c.get("summary", "—")
        if summary:
            pdf.multi_cell(0, 5, f"  Summary: {summary}", new_x="LMARGIN", new_y="NEXT")

    buf = io.BytesIO()
    pdf.output(buf)
    logger.info("PDF report generated: %d candidates, %.1f KB", len(candidates), buf.tell() / 1024)
    return buf.getvalue()


# ── CSV Report ────────────────────────────────────────────────────────────────


def generate_csv(candidates: list[dict]) -> str:
    """
    Generate a CSV string for all candidates.

    Returns:
        UTF-8 CSV string (suitable for st.download_button).
    """
    buf = io.StringIO()
    fieldnames = [
        "rank", "name", "filename", "score", "recommendation",
        "matched_skills", "missing_skills", "bonus_skills", "summary",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for rank, c in enumerate(candidates, 1):
        writer.writerow({
            "rank": rank,
            "name": c.get("name", ""),
            "filename": c.get("filename", ""),
            "score": f"{c.get('score', 0):.2f}",
            "recommendation": c.get("recommendation", ""),
            "matched_skills": ", ".join(c.get("matched_skills", [])),
            "missing_skills": ", ".join(c.get("missing_skills", [])),
            "bonus_skills": ", ".join(c.get("bonus_skills", [])[:10]),
            "summary": c.get("summary", ""),
        })

    logger.info("CSV report generated: %d candidates.", len(candidates))
    return buf.getvalue()
