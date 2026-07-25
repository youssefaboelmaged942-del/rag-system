"""
streamlit_app.py
-----------------
Stage 8 of the RAG pipeline: Streamlit App Interface & Deployment.
"""

import streamlit as st
from importlib import import_module

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Career Advisor", page_icon="🎯", layout="centered")

# --- Background Image Setup ---
bg_image_url = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

st.markdown(f"""
    <style>
    /* استدعاء خط Inter الاحترافي */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* تغيير خلفية التطبيق */
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* رأس الصفحة والعناوين - تصميم بلوري أنيق */
    .stTitle, .stCaption {{
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        display: inline-block;
        margin-bottom: 10px;
    }}

    h1 {{
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }}

    /* تصميم الـ Form وحقول الإدخال */
    [data-testid="stForm"] {{
        background-color: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.2);
    }}

    /* العناوين داخل הـ Form */
    [data-testid="stForm"] label {{
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }}

    /* تحسين زر الإرسال */
    [data-testid="stForm"] button {{
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }}

    [data-testid="stForm"] button:hover {{
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }}

    /* تحسين عرض تقرير النتيجة */
    .stMarkdown h2 {{
        color: #0F172A !important;
        font-weight: 700;
    }}

    /* حاوية نتائج التقرير */
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
        color: #1E293B;
        font-size: 1.02rem;
        line-height: 1.6;
    }}

    /* تحسين القوائم المنسدلة والـ Expander */
    .streamlit-expanderHeader {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-weight: 600 !important;
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
    
    # وضع التقرير داخل بطاقة أنيقة واضحة
    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.93); backdrop-filter: blur(10px); padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            {st.session_state.report}
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.hits:
        st.write("")
        with st.expander("View Retrieved Jobs (Context)"):
            for hit in st.session_state.hits:
                text_content = hit.get('text', str(hit)) if isinstance(hit, dict) else str(hit)
                st.markdown(f"- {text_content}")
