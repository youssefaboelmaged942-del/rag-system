import os
import streamlit as st
from importlib import import_module

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Career Advisor (Egypt)", page_icon="🤖", layout="centered")

# --- Auto-initialize Vector Store for Cloud Deployment ---
_store = import_module("05_create_chroma_store")
build_store = _store.build_store
get_chroma_collection = _store.get_chroma_collection
CHROMA_DIR = _store.CHROMA_DIR

@st.cache_resource
def ensure_vector_store():
    _, collection = get_chroma_collection(CHROMA_DIR)
    if collection.count() == 0:
        with st.spinner("Setting up job database for the first time... This may take a few minutes."):
            build_store()

ensure_vector_store()

# --- Background Image Setup ---
bg_image_url = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp div {{
        color: #FFFFFF !important;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.8);
    }}

    .hero-slogan {{
        font-size: 1.2rem;
        font-weight: 600;
        color: #93C5FD !important;
        margin-top: -10px;
        margin-bottom: 20px;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.8);
    }}

    [data-testid="stForm"] {{
        background-color: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(12px);
        padding: 2rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.5) !important;
    }}

    [data-testid="stForm"] input, [data-testid="stForm"] textarea {{
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        text-shadow: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
    }}

    [data-testid="stForm"] button {{
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
    }}

    [data-testid="stForm"] button:hover {{
        background-color: #1D4ED8 !important;
    }}

    div[data-testid="stExpander"] {{
        background-color: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(12px);
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.5) !important;
    }}

    div[data-testid="stExpander"] summary {{
        border-radius: 16px !important;
    }}

    .report-card ul, .report-card ol {{
        direction: rtl !important;
        text-align: right !important;
        padding-right: 20px !important;
    }}
    </style>
""", unsafe_allow_html=True)

rag_prompting = import_module("07_prompting")
rag_retrieve = import_module("06_retrieve_context")

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

# --- Header & Slogan ---
st.title("🎯 AI Career Advisor (Egypt)")
st.markdown('<p class="hero-slogan">Bridging Skills to Opportunities</p>', unsafe_allow_html=True)
st.caption("Analyze skill gaps and build learning paths for the Egyptian job market using real Wuzzuf jobs.")

# --- Input Form ---
with st.form("career_form"):
    current_skills = st.text_area(
        "Current Skills",
        placeholder="e.g., Python, Excel, SQL, Power BI",
    )
    target_job_title = st.text_input(
        "Target Job Title",
        placeholder="e.g., Data Analyst in Egypt",
    )
    top_k = st.slider("Number of Egyptian jobs to retrieve", min_value=3, max_value=15, value=8)
    submitted = st.form_submit_button("Analyze My Career Path in Egypt")

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
        with st.spinner("Searching Egyptian job market and analyzing skill gap..."):
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
                st.session_state.hits = hits
                st.session_state.report = report
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.session_state.hits = []
                st.session_state.report = None

# --- Display Results ---
if st.session_state.report:
    st.markdown("## 📊 Egypt Career Roadmap & Report")
    
    st.markdown(
        f"""
        <div class="report-card" style="
            background-color: rgba(15, 23, 42, 0.88);
            backdrop-filter: blur(12px);
            padding: 28px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.5);
            color: #FFFFFF;
            line-height: 1.8;
            font-size: 1.05rem;
            margin-bottom: 20px;
            direction: rtl;
            text-align: right;
        ">
            {st.session_state.report}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.hits:
        with st.expander("View Retrieved Egyptian Jobs (Wuzzuf Context)"):
            for hit in st.session_state.hits:
                text_content = hit.get('text', str(hit)) if isinstance(hit, dict) else str(hit)
                st.markdown(f"- {text_content}")
