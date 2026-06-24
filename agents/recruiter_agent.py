"""
agents/recruiter_agent.py — LangGraph 4-agent pipeline (hardened).

Improvements over v1:
  - Prompt truncation to avoid token overflows
  - Structured output validation via Pydantic
  - Graceful fallback when LLM is unavailable
  - Configurable LLM backend (Groq / Ollama / OpenAI)
  - Each node is independently testable (pure function of CandidateState)
"""

from __future__ import annotations

import re
import time
from typing import Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, field_validator
from typing_extensions import TypedDict

from config import get_logger, settings
from utils.cache import cache_agent_result

logger = get_logger(__name__)

MAX_TEXT = settings.llm.max_prompt_chars


# ── State schema ──────────────────────────────────────────────────────────────


class CandidateState(TypedDict):
    name: str
    resume_text: str
    jd_text: str
    matched_skills: list[str]
    missing_skills: list[str]
    skill_assessment: str
    experience_summary: str
    jd_match_analysis: str
    summary: str
    recommendation: Literal["Recommended", "Not Recommended", "Needs Review"]
    errors: list[str]


# ── Pydantic output model ─────────────────────────────────────────────────────


class FinalReport(BaseModel):
    summary: str
    recommendation: Literal["Recommended", "Not Recommended", "Needs Review"]

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Summary is empty")
        return v.strip()

    @field_validator("recommendation")
    @classmethod
    def valid_recommendation(cls, v: str) -> str:
        allowed = {"Recommended", "Not Recommended", "Needs Review"}
        if v not in allowed:
            raise ValueError(f"Recommendation must be one of {allowed}, got {v!r}")
        return v


# ── LLM factory ───────────────────────────────────────────────────────────────


def _get_llm():
    backend = settings.llm.backend
    if backend == "groq":
        from langchain_groq import ChatGroq
        if not settings.llm.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return ChatGroq(
            api_key=settings.llm.groq_api_key,
            model=settings.llm.groq_model,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.llm.ollama_model,
            base_url=settings.embedding.ollama_base_url,
            num_predict=settings.llm.max_tokens,
        )
    elif backend == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.llm.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.openai_model,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r}")


def _call_llm(prompt: str, state: CandidateState) -> str:
    """Call the configured LLM. Returns the text response or an error note."""
    try:
        llm = _get_llm()
        t0 = time.perf_counter()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        logger.debug("LLM call took %.2fs (%d chars)", time.perf_counter() - t0, len(text))
        return text.strip()
    except Exception as exc:
        logger.warning("LLM call failed for '%s': %s", state["name"], exc)
        state["errors"].append(str(exc))
        return f"[LLM unavailable: {exc}]"


def _trunc(text: str, max_chars: int = MAX_TEXT) -> str:
    return text[:max_chars] if len(text) > max_chars else text


# ── Agent nodes ───────────────────────────────────────────────────────────────


def skill_evaluator(state: CandidateState) -> CandidateState:
    """Agent 1 — Technical skill assessment."""
    matched = ", ".join(state["matched_skills"]) or "None"
    missing = ", ".join(state["missing_skills"]) or "None"

    prompt = f"""You are a senior technical recruiter evaluating a candidate's skills.

Candidate: {state['name']}
Matched skills (found in resume): {matched}
Missing skills (required but not found): {missing}

Write a concise technical skills assessment (3–5 sentences) covering:
- Overall technical fit
- Strength of matched skills
- Impact of missing skills
- Whether gaps are learnable

Assessment:"""

    state["skill_assessment"] = _call_llm(prompt, state)
    return state


def experience_evaluator(state: CandidateState) -> CandidateState:
    """Agent 2 — Work experience and project summary."""
    prompt = f"""You are a recruiter summarizing a candidate's experience.

Candidate: {state['name']}
Resume (truncated to first {MAX_TEXT} chars):
{_trunc(state['resume_text'])}

Summarize their work experience and relevant projects in 3–5 sentences.
Focus on: years of experience, domains, key projects, achievements.

Summary:"""

    state["experience_summary"] = _call_llm(prompt, state)
    return state


def jd_matcher(state: CandidateState) -> CandidateState:
    """Agent 3 — Holistic JD alignment analysis."""
    prompt = f"""You are a recruiter assessing overall candidate-JD fit.

Job Description (truncated):
{_trunc(state['jd_text'], 2000)}

Skill Assessment:
{state['skill_assessment']}

Experience Summary:
{state['experience_summary']}

Write a 3–4 sentence holistic analysis comparing this candidate to the JD.
Include: alignment of background, culture/role fit, key strengths, key concerns.

Analysis:"""

    state["jd_match_analysis"] = _call_llm(prompt, state)
    return state


def report_generator(state: CandidateState) -> CandidateState:
    """Agent 4 — Final structured summary and recommendation."""
    prompt = f"""You are writing a final recruitment report. Respond ONLY in this exact format:

SUMMARY: <2-3 sentence candidate summary>
RECOMMENDATION: <Recommended|Not Recommended|Needs Review>

---
Candidate: {state['name']}
Skill Assessment: {state['skill_assessment']}
Experience: {state['experience_summary']}
JD Fit Analysis: {state['jd_match_analysis']}
---

Report:"""

    raw = _call_llm(prompt, state)
    parsed = _parse_final_report(raw, state)
    state["summary"] = parsed.summary
    state["recommendation"] = parsed.recommendation
    return state


def _parse_final_report(raw: str, state: CandidateState) -> FinalReport:
    """Parse and validate agent 4 output. Falls back gracefully on malformed output."""
    summary_match = re.search(r"SUMMARY:\s*(.+?)(?:RECOMMENDATION:|$)", raw, re.DOTALL | re.IGNORECASE)
    rec_match = re.search(r"RECOMMENDATION:\s*(Recommended|Not Recommended|Needs Review)", raw, re.IGNORECASE)

    summary = summary_match.group(1).strip() if summary_match else ""
    recommendation_raw = rec_match.group(1).strip() if rec_match else ""

    # Normalize recommendation
    rec_map = {
        "recommended": "Recommended",
        "not recommended": "Not Recommended",
        "needs review": "Needs Review",
    }
    recommendation = rec_map.get(recommendation_raw.lower(), "Needs Review")

    if not summary:
        logger.warning("Could not parse SUMMARY from agent 4 output for '%s'. Using fallback.", state["name"])
        summary = (
            state.get("jd_match_analysis")
            or state.get("experience_summary")
            or "Summary unavailable — LLM output was malformed."
        )
        state["errors"].append("report_generator: summary parse failed, used fallback.")

    try:
        return FinalReport(summary=summary, recommendation=recommendation)
    except Exception as exc:
        logger.warning("FinalReport validation error for '%s': %s", state["name"], exc)
        state["errors"].append(f"report_generator validation: {exc}")
        return FinalReport(summary=summary or "Parse error.", recommendation="Needs Review")


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    graph = StateGraph(CandidateState)
    graph.add_node("skill_evaluator", skill_evaluator)
    graph.add_node("experience_evaluator", experience_evaluator)
    graph.add_node("jd_matcher", jd_matcher)
    graph.add_node("report_generator", report_generator)

    graph.set_entry_point("skill_evaluator")
    graph.add_edge("skill_evaluator", "experience_evaluator")
    graph.add_edge("experience_evaluator", "jd_matcher")
    graph.add_edge("jd_matcher", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@cache_agent_result
def run_agent_pipeline(
    name: str,
    resume_text: str,
    jd_text: str,
    matched_skills: list[str],
    missing_skills: list[str],
) -> CandidateState:
    """
    Run the full 4-agent pipeline for one candidate.

    Returns a CandidateState dict with all fields populated.
    Errors (if any) are collected in state["errors"], never raised.
    """
    initial_state: CandidateState = {
        "name": name,
        "resume_text": resume_text,
        "jd_text": jd_text,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_assessment": "",
        "experience_summary": "",
        "jd_match_analysis": "",
        "summary": "",
        "recommendation": "Needs Review",
        "errors": [],
    }

    try:
        graph = get_graph()
        result = graph.invoke(initial_state)
        return result
    except Exception as exc:
        logger.exception("Pipeline failed for candidate '%s'", name)
        initial_state["errors"].append(f"Pipeline error: {exc}")
        initial_state["summary"] = "Analysis failed. Please retry."
        initial_state["recommendation"] = "Needs Review"
        return initial_state
