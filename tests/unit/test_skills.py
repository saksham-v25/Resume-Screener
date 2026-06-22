"""
tests/unit/test_skills.py — Unit tests for utils/skills.py
"""

from __future__ import annotations

import pytest


class TestExtractSkills:
    def test_extracts_python(self):
        from utils.skills import extract_skills
        skills = extract_skills("Experienced with Python and machine learning.")
        assert "Python" in skills

    def test_extracts_alias(self):
        from utils.skills import extract_skills
        # "sklearn" should map to "scikit-learn"
        skills = extract_skills("Used sklearn and torch for model training.")
        assert "scikit-learn" in skills
        assert "PyTorch" in skills

    def test_case_insensitive(self):
        from utils.skills import extract_skills
        assert "Python" in extract_skills("PYTHON developer")
        assert "Docker" in extract_skills("docker containerization")

    def test_no_false_positives(self):
        from utils.skills import extract_skills
        # "r" should not match "R" language in a sentence full of words
        skills = extract_skills("The car drove far and hard.")
        assert "R" not in skills

    def test_empty_text(self):
        from utils.skills import extract_skills
        assert extract_skills("") == set()

    def test_multiword_skill(self):
        from utils.skills import extract_skills
        assert "Machine Learning" in extract_skills("5 years of machine learning experience")

    def test_synonym_maps_to_canonical(self):
        from utils.skills import extract_skills
        # "pyspark" → "Apache Spark"
        assert "Apache Spark" in extract_skills("Built ETL pipelines using PySpark")


class TestAnalyzeSkills:
    def test_matched_and_missing(self):
        from utils.skills import analyze_skills
        jd = "Requires Python, Docker, and AWS experience."
        resume = "Proficient in Python and Docker."
        result = analyze_skills(resume, jd)
        assert "Python" in result.matched
        assert "Docker" in result.matched
        assert "AWS" in result.missing

    def test_bonus_skills(self):
        from utils.skills import analyze_skills
        jd = "Requires Python."
        resume = "Expert in Python, Kubernetes, and Rust."
        result = analyze_skills(resume, jd)
        assert "Python" in result.matched
        # Kubernetes and/or Rust should appear as bonus
        assert len(result.bonus) >= 1

    def test_empty_jd(self):
        from utils.skills import analyze_skills
        result = analyze_skills("Python expert", "")
        assert result.matched == []
        assert result.missing == []

    def test_empty_resume(self):
        from utils.skills import analyze_skills
        result = analyze_skills("", "Requires Python and AWS.")
        assert result.matched == []
        assert len(result.missing) >= 2

    def test_no_duplicates_in_matched(self):
        from utils.skills import analyze_skills
        # "Python" appears multiple times in JD
        jd = "Python Python Python developer"
        resume = "Python expert"
        result = analyze_skills(resume, jd)
        assert result.matched.count("Python") == 1

    def test_returns_sorted_lists(self):
        from utils.skills import analyze_skills
        jd = "Python, Docker, AWS, Kubernetes, Go"
        resume = "Python, Docker"
        result = analyze_skills(resume, jd)
        assert result.matched == sorted(result.matched)
        assert result.missing == sorted(result.missing)
