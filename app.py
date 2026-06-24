"""
app.py — AI Recruiter Assistant (Industry-hardened edition)

Tabs:
  1. Upload      — Parse JD + resumes, trigger full analysis
  2. Ranking     — Scored candidate cards with skill charts
  3. Reports     — PDF/CSV export + side-by-side comparison
  4. Search      — Semantic FAISS-powered query
  5. AI Chat     — Groq-powered Q&A about candidates
  6. Status      — System health, logs, telemetry
  7. History     — Past analysis runs (SQLite)
"""

from __future__ import annotations

import io
import os
import traceback

import streamlit as st

from config import get_logger, settings, setup_logging

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging(
    settings.log_level,
    settings.log_dir,
    json_output=os.getenv("LOG_FORMAT", "text") == "json",
)
logger = get_logger(__name__)

# ── Startup services (telemetry + persistence) ────────────────────────────────
from utils.database import init_db  # noqa: E402
from utils.telemetry import init_telemetry_db, track_event  # noqa: E402

init_telemetry_db()
init_db()

st.set_page_config(
    page_title="AI Recruiter Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "candidates": [],
    "jd_text": "",
    "analysis_done": False,
    "analysis_errors": [],
    "analysis_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption(f"Embedding backend: **{settings.embedding.backend}**")
    st.caption(f"LLM backend: **{settings.llm.backend}**")
    st.caption(f"Cache: **{'ON' if settings.cache.enabled else 'OFF'}**")

    if st.button("🔄 Reset session"):
        for key in ["candidates", "jd_text", "analysis_done", "analysis_errors", "analysis_log"]:
            st.session_state[key] = [] if key != "jd_text" and key != "analysis_done" else (
                "" if key == "jd_text" else False
            )
        from vectorstore.store import invalidate_index
        invalidate_index()
        st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📤 Upload", "🏆 Ranking", "📄 Reports", "🔍 Search", "💬 AI Chat", "🩺 Status", "📚 History"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Upload Job Description & Resumes")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Description")
        jd_input_method = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True)

        if jd_input_method == "Paste text":
            jd_text = st.text_area("Paste JD here", height=300, placeholder="Paste the full job description…")
        else:
            jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], key="jd_file")
            jd_text = ""
            if jd_file:
                from utils.parser import parse_resume
                parsed_jd = parse_resume(jd_file.read(), jd_file.name)
                if parsed_jd["parse_error"]:
                    st.error(f"❌ Could not parse JD: {parsed_jd['parse_error']}")
                else:
                    jd_text = parsed_jd["text"]
                    st.success(f"✅ JD loaded ({len(jd_text):,} chars)")

    with col2:
        st.subheader("Resumes")
        resume_files = st.file_uploader(
            "Upload one or more resume PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

    # Validation preview
    if jd_text and resume_files:
        from utils.validators import validate_jd
        jd_ok, jd_err = validate_jd(jd_text)
        if not jd_ok:
            st.error(f"❌ Job Description: {jd_err}")
        else:
            st.info(f"Ready: **{len(resume_files)}** resumes vs JD ({len(jd_text):,} chars)")
    elif not jd_text:
        st.warning("Please provide a job description.")
    elif not resume_files:
        st.warning("Please upload at least one resume PDF.")

    if st.button("🚀 Run Full Analysis", type="primary", disabled=not (jd_text and resume_files)):
        st.session_state["jd_text"] = jd_text
        st.session_state["analysis_errors"] = []
        st.session_state["analysis_log"] = []
        st.session_state["analysis_done"] = False
        candidates_out = []

        progress = st.progress(0, text="Parsing resumes…")
        status_box = st.empty()
        log_lines: list[str] = []

        try:
            # Step 1: Parse resumes
            from utils.parser import parse_resume
            parsed_resumes = []
            for i, rf in enumerate(resume_files):
                progress.progress((i + 1) / (len(resume_files) * 4), text=f"Parsing {rf.name}…")
                pr = parse_resume(rf.read(), rf.name)
                if pr["parse_error"]:
                    msg = f"⚠️ {rf.name}: {pr['parse_error']}"
                    st.session_state["analysis_errors"].append(msg)
                    log_lines.append(msg)
                else:
                    parsed_resumes.append(pr)

            if not parsed_resumes:
                st.error("❌ No resumes could be parsed. Please check your PDF files.")
                st.stop()

            # Step 2: Rank
            progress.progress(0.35, text="Ranking candidates…")
            from utils.ranking import rank_candidates
            try:
                ranked = rank_candidates(parsed_resumes, jd_text)
            except RuntimeError as exc:
                st.error(f"❌ Ranking failed: {exc}")
                st.info("💡 Check your embedding backend in the sidebar.")
                st.stop()

            # Step 3: Skill analysis
            progress.progress(0.50, text="Extracting skills…")
            from utils.skills import analyze_skills
            for r in ranked:
                skills = analyze_skills(r["text"], jd_text)
                r["matched_skills"] = skills.matched
                r["missing_skills"] = skills.missing
                r["bonus_skills"] = skills.bonus

            # Step 4: Build FAISS index
            progress.progress(0.60, text="Building semantic index…")
            from vectorstore.store import build_index
            try:
                build_index([{"name": r["name"], "filename": r["filename"], "text": r["text"]} for r in ranked])
            except Exception as exc:
                msg = f"⚠️ FAISS index build failed (semantic search disabled): {exc}"
                st.session_state["analysis_errors"].append(msg)

            # Step 5: Run AI agents
            from agents.recruiter_agent import run_agent_pipeline
            n = len(ranked)
            for i, r in enumerate(ranked):
                progress.progress(0.65 + 0.33 * (i / n), text=f"Analysing {r['name']} ({i+1}/{n})…")
                result = run_agent_pipeline(
                    name=r["name"],
                    resume_text=r["text"],
                    jd_text=jd_text,
                    matched_skills=r["matched_skills"],
                    missing_skills=r["missing_skills"],
                )
                r["summary"] = result["summary"]
                r["recommendation"] = result["recommendation"]
                r["skill_assessment"] = result["skill_assessment"]
                r["experience_summary"] = result["experience_summary"]
                r["jd_match_analysis"] = result["jd_match_analysis"]
                r["agent_errors"] = result["errors"]

                if result["errors"]:
                    for err in result["errors"]:
                        st.session_state["analysis_errors"].append(f"{r['name']}: {err}")

                candidates_out.append(r)
                log_lines.append(f"✅ {r['name']} → {r['recommendation']} (score: {r['score']:.1f}%)")

            progress.progress(1.0, text="Done!")
            st.session_state["candidates"] = candidates_out
            st.session_state["analysis_done"] = True
            st.session_state["analysis_log"] = log_lines

            # ── Persist to SQLite ──────────────────────────────────────────
            from utils.database import save_analysis
            save_analysis(jd_text, candidates_out)

            # ── Telemetry ────────────────────────────────────────────────────
            track_event(
                "analysis_complete",
                {
                    "candidate_count": len(candidates_out),
                    "error_count": len(st.session_state["analysis_errors"]),
                },
            )

            st.success(f"✅ Analysis complete: {len(candidates_out)} candidates ranked.")
            if st.session_state["analysis_errors"]:
                st.warning(f"⚠️ {len(st.session_state['analysis_errors'])} warning(s) — see **Status** tab.")

        except Exception as exc:
            logger.exception("Analysis pipeline crashed")
            st.error(f"❌ Unexpected error: {exc}")
            st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Ranking
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("🏆 Candidate Rankings")

    if not st.session_state["analysis_done"]:
        st.info("Run analysis in the **Upload** tab first.")
    else:
        candidates = st.session_state["candidates"]
        st.caption(f"{len(candidates)} candidates analysed against your JD")

        # Summary bar chart
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=[c["score"] for c in candidates],
            y=[c["name"] for c in candidates],
            orientation="h",
            marker_color=[
                "#2ecc71" if c["recommendation"] == "Recommended"
                else "#e74c3c" if c["recommendation"] == "Not Recommended"
                else "#f39c12"
                for c in candidates
            ],
        ))
        fig.update_layout(
            title="Match Score by Candidate",
            xaxis_title="Similarity Score (%)",
            yaxis_title="",
            height=max(300, len(candidates) * 45),
            xaxis_range=[0, 100],
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Candidate cards
        for rank, c in enumerate(candidates, 1):
            badge = {
                "Recommended": "🟢",
                "Not Recommended": "🔴",
                "Needs Review": "🟡",
            }.get(c["recommendation"], "⚪")

            with st.expander(f"{rank}. {badge} **{c['name']}** — {c['score']:.1f}%  |  {c['recommendation']}", expanded=(rank == 1)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**📝 Summary**")
                    st.write(c.get("summary", "—"))

                    st.markdown("**🧠 Skill Assessment**")
                    st.write(c.get("skill_assessment", "—"))

                    st.markdown("**💼 Experience**")
                    st.write(c.get("experience_summary", "—"))

                with col2:
                    st.markdown("**✅ Matched Skills**")
                    for s in c.get("matched_skills", []):
                        st.markdown(f"- {s}")

                    st.markdown("**❌ Missing Skills**")
                    missing = c.get("missing_skills", [])
                    if missing:
                        for s in missing:
                            st.markdown(f"- {s}")
                    else:
                        st.markdown("- None")

                    st.markdown("**⭐ Bonus Skills**")
                    for s in c.get("bonus_skills", [])[:8]:
                        st.markdown(f"- {s}")

                if c.get("agent_errors"):
                    with st.expander("⚠️ Agent warnings"):
                        for err in c["agent_errors"]:
                            st.caption(err)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Reports
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("📄 Reports")

    if not st.session_state["analysis_done"]:
        st.info("Run analysis first.")
    else:
        candidates = st.session_state["candidates"]
        jd_text = st.session_state["jd_text"]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download PDF Report"):
                try:
                    from utils.report import generate_pdf
                    pdf_bytes = generate_pdf(candidates, jd_text)
                    st.download_button("Save PDF", pdf_bytes, "recruitment_report.pdf", "application/pdf")
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")

        with col2:
            if st.button("📊 Download CSV"):
                import csv
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=["name", "filename", "score", "recommendation",
                                                          "matched_skills", "missing_skills"])
                writer.writeheader()
                for c in candidates:
                    writer.writerow({
                        "name": c["name"],
                        "filename": c["filename"],
                        "score": c["score"],
                        "recommendation": c["recommendation"],
                        "matched_skills": ", ".join(c.get("matched_skills", [])),
                        "missing_skills": ", ".join(c.get("missing_skills", [])),
                    })
                st.download_button("Save CSV", buf.getvalue(), "candidates.csv", "text/csv")

        # Side-by-side comparison
        if len(candidates) >= 2:
            st.subheader("Side-by-side comparison")
            names = [c["name"] for c in candidates]
            sel1 = st.selectbox("Candidate A", names, index=0, key="cmp_a")
            sel2 = st.selectbox("Candidate B", names, index=1, key="cmp_b")
            a = next(c for c in candidates if c["name"] == sel1)
            b = next(c for c in candidates if c["name"] == sel2)

            col_a, col_b = st.columns(2)
            for col, cand in [(col_a, a), (col_b, b)]:
                with col:
                    st.markdown(f"### {cand['name']}")
                    st.metric("Match Score", f"{cand['score']:.1f}%")
                    st.metric("Recommendation", cand["recommendation"])
                    st.write("**Matched:**", ", ".join(cand.get("matched_skills", [])) or "None")
                    st.write("**Missing:**", ", ".join(cand.get("missing_skills", [])) or "None")
                    st.write("**Summary:**", cand.get("summary", "—"))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Semantic Search
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("🔍 Semantic Search")
    st.caption("Find candidates using natural language — e.g. 'Python developer with NLP experience'")

    from vectorstore.store import index_is_built
    from vectorstore.store import search as faiss_search

    if not index_is_built():
        st.info("Run analysis first to build the search index.")
    else:
        query = st.text_input("Search query")
        top_k = st.slider("Results", 1, 10, 5)
        if query and st.button("Search"):
            try:
                results = faiss_search(query, top_k=top_k)
                if not results:
                    st.warning("No results found.")
                for r in results:
                    with st.expander(f"**{r['name']}** — Similarity: {r['similarity_score']:.1f}%"):
                        st.write(r["text"][:800] + "…")
            except Exception as exc:
                st.error(f"Search failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — AI Chat
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("💬 AI Chat")

    if not st.session_state["analysis_done"]:
        st.info("Run analysis first to enable AI chat.")
    else:
        candidates = st.session_state["candidates"]
        context = "\n\n".join(
            f"Candidate: {c['name']}\nScore: {c['score']:.1f}%\nRecommendation: {c['recommendation']}\n"
            f"Summary: {c.get('summary', '')}"
            for c in candidates
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about candidates…")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            try:
                from agents.recruiter_agent import _get_llm
                llm = _get_llm()
                prompt = (
                    f"You are an AI recruiting assistant. Here are the candidates:\n\n{context}\n\n"
                    f"User question: {user_input}\n\nAnswer helpfully and concisely."
                )
                response = llm.invoke(prompt)
                reply = response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                reply = f"⚠️ Chat unavailable: {exc}"

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Status
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("🩺 System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Embedding Backend")
        backend = settings.embedding.backend
        if backend == "ollama":
            from utils.embeddings import check_ollama_health
            healthy, msg = check_ollama_health()
            if healthy:
                st.success(f"✅ Ollama: {msg}")
            else:
                st.error(f"❌ Ollama: {msg}")
                st.info("💡 Set `EMBEDDING_BACKEND=huggingface` to use a local model with no server.")
        else:
            st.success("✅ HuggingFace (local — no server needed)")

        st.subheader("LLM Backend")
        llm_backend = settings.llm.backend
        if llm_backend == "groq":
            if settings.llm.groq_api_key:
                st.success("✅ Groq API key found")
            else:
                st.error("❌ GROQ_API_KEY not set")
        elif llm_backend == "openai":
            if settings.llm.openai_api_key:
                st.success("✅ OpenAI API key found")
            else:
                st.error("❌ OPENAI_API_KEY not set")
        else:
            st.info(f"LLM backend: {llm_backend}")

    with col2:
        st.subheader("FAISS Index")
        from vectorstore import store as _vs
        if _vs.index_is_built():
            st.success(f"✅ Index built ({_vs._index.ntotal} vectors)")
        else:
            st.warning("⚠️ Index not built (run analysis first)")

        st.subheader("Configuration")
        st.json({
            "embedding_backend": settings.embedding.backend,
            "llm_backend": settings.llm.backend,
            "cache_enabled": settings.cache.enabled,
            "skill_use_embeddings": settings.skill.use_embeddings,
            "fuzzy_threshold": settings.skill.fuzzy_threshold,
        })

    if st.session_state["analysis_errors"]:
        st.subheader("⚠️ Warnings / Errors")
        for err in st.session_state["analysis_errors"]:
            st.warning(err)

    if st.session_state["analysis_log"]:
        st.subheader("📋 Analysis Log")
        st.code("\n".join(st.session_state["analysis_log"]))

    st.subheader("📡 Recent Telemetry Events")
    from utils.telemetry import get_recent_events
    events = get_recent_events(limit=15)
    if not events:
        st.caption("No telemetry events yet — run an analysis first.")
    else:
        for ev in events:
            st.caption(
                f"`{ev['timestamp'][:19]}` &nbsp; **{ev['event_type']}** &nbsp; "
                f"{ev['duration_ms']:.0f} ms &nbsp; `{ev['metadata']}`"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — History
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("📚 Analysis History")
    st.caption("All previous analysis runs — stored locally in SQLite.")

    from utils.database import get_analysis_history, get_candidates_by_analysis_id

    if st.button("🔄 Refresh", key="history_refresh"):
        st.rerun()

    history = get_analysis_history(limit=25)

    if not history:
        st.info("No previous analyses found. Run your first analysis in the **Upload** tab!")
    else:
        for run in history:
            badge_map = {
                "Recommended":     "🟢",
                "Not Recommended": "🔴",
                "Needs Review":    "🟡",
            }
            created = run["created_at"][:19].replace("T", " ")
            label = (
                f"📅 {created} — {run['candidate_count']} candidates — "
                f"{run['jd_snippet'][:60]}{'...' if len(run['jd_snippet']) > 60 else ''}"
            )
            with st.expander(label):
                candidates_hist = get_candidates_by_analysis_id(run["id"])
                if not candidates_hist:
                    st.caption("No candidate data found.")
                    continue

                # Summary table
                col_n, col_s, col_r = st.columns([3, 1, 2])
                col_n.markdown("**Candidate**")
                col_s.markdown("**Score**")
                col_r.markdown("**Recommendation**")
                st.divider()

                for cand in candidates_hist:
                    badge = badge_map.get(cand["recommendation"], "⚪")
                    c1, c2, c3 = st.columns([3, 1, 2])
                    c1.write(cand["name"])
                    c2.write(f"{cand['score']:.1f}%")
                    c3.write(f"{badge} {cand['recommendation']}")

                if st.checkbox("Show summaries", key=f"hist_sum_{run['id']}"):
                    for cand in candidates_hist:
                        st.markdown(f"**{cand['name']}:** {cand['summary'] or '—'}")

