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

# --- تهيئة Session State لحفظ النتائج بين دورات اعادة التشغيل (Reruns) ---
if "report" not in st.session_state:
    st.session_state.report = None
if "hits" not in st.session_state:
    st.session_state.hits = []

st.title("🎯 AI Career Advisor")
st.caption("Analyze skill gaps and build practical learning roadmaps powered by real-time Wuzzuf job data.")

with st.form("career_form"):
    current_skills = st.text_area(
        "المهارات الحالية (current skills)",
        placeholder="مثال: Python, Excel, SQL, Power BI",
    )
    target_job_title = st.text_input(
        "المسمى الوظيفي المستهدف (target job title)",
        placeholder="مثال: Data Analyst",
    )
    top_k = st.slider("عدد الوظائف المرجعية لاسترجاعها", min_value=3, max_value=15, value=8)
    submitted = st.form_submit_button("حلل مساري المهني")

# --- عند النقر على الزر يتم جلب البيانات وحفظها في session_state ---
if submitted:
    if not current_skills.strip() or not target_job_title.strip():
        st.warning("من فضلك أدخل كلًا من المهارات الحالية والمسمى الوظيفي المستهدف.")
    elif not rag_prompting.OPENROUTER_API_KEY:
        st.error(
            "لم يتم العثور على مفتاح OpenRouter API. من فضلك أضفه في Streamlit "
            "Secrets (OPENROUTER_API_KEY) قبل التشغيل."
        )
    else:
        with st.spinner("جاري البحث في سوق العمل وتحليل الفجوة..."):
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
                # حفظ النتائج في الـ State
                st.session_state.hits = hits
                st.session_state.report = report
            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {e}")
                st.session_state.hits = []
                st.session_state.report = None

# --- عرض التقرير والـ Expander بشكل مستقل بناءً على session_state ---
if st.session_state.report:
    st.markdown("## 📊 التقرير المهني")
    st.markdown(st.session_state.report)

    if st.session_state.hits:
        with st.expander("عرض الوظائف المسترجعة من قاعدة البيانات (context)"):
            for hit in st.session_state.hits:
                # التحقق من بناء القاموس (dict أو object) لعرض النص ب أمان
                text_content = hit.get('text', str(hit)) if isinstance(hit, dict) else str(hit)
                st.markdown(f"- {text_content}")
