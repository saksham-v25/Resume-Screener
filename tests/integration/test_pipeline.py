"""
tests/integration/test_pipeline.py — Integration tests for the recruiter agent pipeline.

These tests mock LLM calls but exercise the full LangGraph graph execution.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


SAMPLE_JD = """
We are looking for a Senior Python Developer with experience in:
- Python (5+ years)
- AWS and Docker for deployment
- Machine Learning with scikit-learn or PyTorch
- REST API development with FastAPI
- PostgreSQL / SQL databases
"""

SAMPLE_RESUME = """
Jane Doe
Senior Software Engineer

Experience:
- 6 years of Python development
- Built ML pipelines using scikit-learn and PyTorch
- Deployed microservices on AWS using Docker and Kubernetes
- Developed REST APIs with FastAPI and Flask
- PostgreSQL and Redis for data storage

Education: B.Tech Computer Science
"""


def _mock_llm_response(content: str):
    """Create a mock LangChain LLM response object."""
    mock = MagicMock()
    mock.content = content
    return mock


class TestAgentPipeline:
    @patch("agents.recruiter_agent._get_llm")
    def test_full_pipeline_runs(self, mock_get_llm):
        """Full pipeline completes without exception and returns all fields."""
        mock_llm = MagicMock()

        # Return appropriate mock responses for each agent call
        responses = [
            _mock_llm_response("Strong technical skills with Python and ML."),
            _mock_llm_response("6 years of Python, ML pipelines, cloud deployment."),
            _mock_llm_response("Excellent fit for the role. Strong alignment with JD."),
            _mock_llm_response(
                "SUMMARY: Jane is a highly qualified Python developer with ML expertise.\n"
                "RECOMMENDATION: Recommended"
            ),
        ]
        mock_llm.invoke.side_effect = responses
        mock_get_llm.return_value = mock_llm

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline(
            name="Jane Doe",
            resume_text=SAMPLE_RESUME,
            jd_text=SAMPLE_JD,
            matched_skills=["Python", "AWS", "Docker", "scikit-learn"],
            missing_skills=["Kubernetes"],
        )

        assert result["name"] == "Jane Doe"
        assert result["recommendation"] in {"Recommended", "Not Recommended", "Needs Review"}
        assert len(result["summary"]) > 0
        assert len(result["skill_assessment"]) > 0
        assert len(result["experience_summary"]) > 0
        assert len(result["jd_match_analysis"]) > 0

    @patch("agents.recruiter_agent._get_llm")
    def test_malformed_report_falls_back(self, mock_get_llm):
        """When agent 4 gives malformed output, fallback summary is used."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _mock_llm_response("Skills ok."),
            _mock_llm_response("Experienced dev."),
            _mock_llm_response("Good JD fit."),
            _mock_llm_response("This is totally unstructured output with no SUMMARY/RECOMMENDATION"),
        ]
        mock_get_llm.return_value = mock_llm

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline("Bob", SAMPLE_RESUME, SAMPLE_JD, ["Python"], ["AWS"])

        # Should not crash; should have a fallback summary
        assert len(result["summary"]) > 0
        assert result["recommendation"] in {"Recommended", "Not Recommended", "Needs Review"}
        # Error should be logged
        assert any("summary parse failed" in e or "fallback" in e.lower() for e in result["errors"])

    @patch("agents.recruiter_agent._get_llm")
    def test_llm_unavailable_returns_graceful_state(self, mock_get_llm):
        """When LLM is completely unavailable, pipeline returns a Needs Review state."""
        mock_get_llm.side_effect = RuntimeError("GROQ_API_KEY not set")

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline("Carol", SAMPLE_RESUME, SAMPLE_JD, [], [])

        assert result["recommendation"] == "Needs Review"
        assert len(result["errors"]) > 0

    @patch("agents.recruiter_agent._get_llm")
    def test_recommended_parsing(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _mock_llm_response("Good skills."),
            _mock_llm_response("Solid experience."),
            _mock_llm_response("Strong fit."),
            _mock_llm_response("SUMMARY: Outstanding candidate.\nRECOMMENDATION: Recommended"),
        ]
        mock_get_llm.return_value = mock_llm

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline("Dan", SAMPLE_RESUME, SAMPLE_JD, ["Python"], [])
        assert result["recommendation"] == "Recommended"

    @patch("agents.recruiter_agent._get_llm")
    def test_not_recommended_parsing(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _mock_llm_response("Weak skills."),
            _mock_llm_response("Limited experience."),
            _mock_llm_response("Poor fit."),
            _mock_llm_response("SUMMARY: Not a good fit.\nRECOMMENDATION: Not Recommended"),
        ]
        mock_get_llm.return_value = mock_llm

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline("Eve", SAMPLE_RESUME, SAMPLE_JD, [], ["Python", "AWS"])
        assert result["recommendation"] == "Not Recommended"


class TestOutputFormatting:
    """Validate that pipeline outputs can be consumed downstream without errors."""

    @patch("agents.recruiter_agent._get_llm")
    def test_state_fields_are_strings(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(
            "SUMMARY: Good candidate.\nRECOMMENDATION: Recommended"
        )
        mock_get_llm.return_value = mock_llm

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline("Frank", SAMPLE_RESUME, SAMPLE_JD, ["Python"], [])

        for field in ("name", "summary", "recommendation", "skill_assessment",
                      "experience_summary", "jd_match_analysis"):
            assert isinstance(result[field], str), f"{field} should be a string"

    @patch("agents.recruiter_agent._get_llm")
    def test_errors_is_list(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(
            "SUMMARY: Ok.\nRECOMMENDATION: Needs Review"
        )
        mock_get_llm.return_value = mock_llm

        from agents.recruiter_agent import run_agent_pipeline
        result = run_agent_pipeline("Grace", SAMPLE_RESUME, SAMPLE_JD, [], [])
        assert isinstance(result["errors"], list)
