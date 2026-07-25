"""
streamlit_app.py
-----------------
Stage 8 of the RAG pipeline: Streamlit App Interface & Deployment.
"""

import streamlit as st
from importlib import import_module

rag_prompting = import_module("07_prompting")
rag_retrieve = import_module("06_retrieve_context")

st.set_page_config(page_title="AI Career Advisor", page_icon="🎯", layout="centered")

# --- Secure API key loading ---
try:
    if not rag_prompting.OPENROUTER_API_KEY:
        rag_prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag_prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag_prompting.OPENROUTER_MODEL)
except Exception:
    pass

# --- Initialize Session State ---
if "report" not in st.session_state:
    st.session_state.report = None
if "hits" not in st.session_state:
    st.session_state.hits = []

# --- Header ---
st.title("🎯 AI Career Advisor")
st.caption("Analyze skill gaps and build learning paths using real Wuzzuf jobs.")

# --- Input Form ---
with st.form("career_form"):
    current_skills = st.text_area(
        "Current Skills",
        placeholder="e.g., Python, Excel, SQL, Power BI",
    )
    target_job_title = st.text_input(
        "Target Job Title",
        placeholder="e.g., Data Analyst",
    )
    top_k = st.slider("Number of jobs to retrieve", min_value=3, max_value=15, value=8)
    submitted = st.form_submit_button("Analyze My Career Path")

# --- Form Submission Handling ---
if submitted:
    if not current_skills.strip() or not target_job_title.strip():
        st.warning("Please enter both your current skills and target job title.")
    elif not rag_prompting.OPENROUTER_API_KEY:
        st.error(
            "OpenRouter API key not found. Please add OPENROUTER_API_KEY "
            "to Streamlit Secrets before running."
        )
    else:
        with st.spinner("Searching job market and analyzing skill gap..."):
            try:
                hits = rag_retrieve.retrieve_context(
                    current_skills=current_skills,
                    target_job_title=target_job_title,
                    top_k=top_k,
                )
                report = rag_prompting.generate_roadmap(
                    current_skills=current_skills,
                    target_job_title=target_job_title,
                    retrieved_context=hits,
                )
                # Save to session state
                st.session_state.hits = hits
                st.session_state.report = report
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.session_state.hits = []
                st.session_state.report = None

# --- Display Results ---
if st.session_state.report:
    st.markdown("## 📊 Career Report")
    st.markdown(st.session_state.report)

    if st.session_state.hits:
        with st.expander("View Retrieved Jobs (Context)"):
            for hit in st.session_state.hits:
                text_content = hit.get('text', str(hit)) if isinstance(hit, dict) else str(hit)
                st.markdown(f"- {text_content}")
