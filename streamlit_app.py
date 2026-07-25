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

    /* رأس الصفحة والعناوين - تصميم بلوري أنيق لحل مشكلة الوضوح */
    .stTitle, .stCaption {{
        background: rgba(255, 255, 255, 0.85); /* خلفية بيضاء شفافة */
        backdrop-filter: blur(10px);          /* تأثير بلوري خفيف */
        padding: 20px 25px;                  /* تباعد داخلي أكبر */
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        display: inline-block;               /* جعل الحاوية تتكيف مع النص */
        margin-bottom: 10px;
    }}

    /* جعل العنوان والنص الوصفي أغمق للتباين */
    h1#ai-career-advisor {{
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }}

    /* جعل النص الوصفي أغمق للوضوح التام */
    .stCaption p {{
        color: #1E293B !important;
        font-size: 1.05rem !important;
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

    /* العناوين داخل الـ Form */
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

    </style>
""", unsafe_allow_html=True)

# ... باقي كود التطبيق (الاستيراد، المنطق، إلخ) يبقى كما هو ...
rag_prompting = import_module("07_prompting")
# ...
