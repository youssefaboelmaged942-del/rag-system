"""
streamlit_app.py
-----------------
Stage 8 of the RAG pipeline: Streamlit App Interface & Deployment.

A simple Streamlit UI for the AI Career Advisor:
  - Text inputs for 'Current Skills' and 'Target Job Title'.
  - Retrieves relevant real job postings from the Wuzzuf vector store.
  - Calls the OpenRouter LLM to generate a structured Arabic report:
      المهارات المتوفرة / فجوة المهارات / خطة التعلم / المصادر والمراجع

API key handling:
  The real OpenRouter API key is never written into this file. When
  deployed on Streamlit Cloud, it is read securely from
  st.secrets (configured under "Manage app" -> "Secrets").
"""

import streamlit as st
from importlib import import_module

rag_prompting = import_module("07_prompting")
rag_retrieve = import_module("06_retrieve_context")

st.set_page_config(page_title="مستشار المسار المهني الذكي", page_icon="🎯", layout="centered")

# --- Secure API key loading: prefer Streamlit secrets when deployed ---
try:
    if not rag_prompting.OPENROUTER_API_KEY:
        rag_prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag_prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag_prompting.OPENROUTER_MODEL)
except Exception:
    # st.secrets raises if no secrets.toml exists (e.g. local run without
    # secrets configured) -- fall back silently to env vars already set
    # on the module.
    pass

st.title("🎯 مستشار المسار المهني الذكي")
st.caption("مبني على وظائف حقيقية من Wuzzuf لتحليل فجوة المهارات وبناء خطة تعلم عملية.")

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
            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {e}")
                hits, report = [], None

        if report:
            st.markdown("## 📊 التقرير المهني")
            st.markdown(report)

            with st.expander("عرض الوظائف المسترجعة من قاعدة البيانات (context)"):
                for hit in hits:
                    st.markdown(f"- {hit['text']}")

st.divider()
st.caption(
    "المشروع يتبع تسلسل المعمل: documents -> preprocessing -> chunking -> "
    "vector representation -> vector store -> context retrieval -> prompting -> Streamlit UI."
)
