"""
07_prompting.py
----------------
Stage 7 of the RAG pipeline: Prompt Engineering & OpenRouter LLM Call.

Builds a structured prompt combining the retrieved job-market context
with the user's current skills and target job title, then sends it to
an LLM via the OpenRouter API to produce a structured Arabic career
report with:
  a) Matching skills (المهارات المتوفرة)
  b) Skill gap analysis (فجوة المهارات)
  c) Career roadmap (خطة التعلم)
  d) Citations (المصادر والمراجع) -- referencing real company names
     and job titles from the retrieved context.

API Key rules (see project instructions):
  - Never hard-code the real API key in this file.
  - OPENROUTER_API_KEY / OPENROUTER_MODEL are read from the
    environment first (useful for local dev with a .env file loaded
    by the shell / python-dotenv) and can be overridden by
    Streamlit secrets at runtime (see streamlit_app.py).
"""

import os
import requests

# These are placeholders only. Real values must come from environment
# variables or Streamlit secrets -- never hard-code a real key here.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an expert AI Career Advisor specializing in the Egyptian job "
    "market. You are given real job postings retrieved from Wuzzuf as "
    "context. Use ONLY this context to judge what skills the market "
    "demands -- do not invent companies or job titles that are not in the "
    "context. Some context items include a real listing URL after "
    "'Source:' -- when citing those specific postings in the "
    "'المصادر والمراجع' section, include the URL alongside the company/"
    "title so the user can verify it directly; for items without a URL, "
    "cite the company name and job title only. Always respond in Arabic, "
    "structured in exactly four sections using these headers: "
    "'المهارات المتوفرة', 'فجوة المهارات', 'خطة التعلم', 'المصادر والمراجع'."
)


def build_prompt(current_skills: str, target_job_title: str, retrieved_context: list) -> str:
    """Construct the user-turn prompt combining retrieved context with
    the user's profile."""
    context_block = "\n".join(f"- {hit['text']}" for hit in retrieved_context) or "لا يوجد سياق متاح."

    prompt = f"""
بيانات المستخدم:
- المسمى الوظيفي المستهدف: {target_job_title}
- المهارات الحالية: {current_skills}

سياق سوق العمل (وظائف حقيقية من Wuzzuf):
{context_block}

المطلوب:
اكتب تقريراً مهنياً منظماً باللغة العربية يحتوي على أربعة أقسام بالضبط،
بنفس هذه العناوين:

1) المهارات المتوفرة: اذكر المهارات التي يمتلكها المستخدم بالفعل وتتوافق
   مع متطلبات السوق الظاهرة في السياق أعلاه.
2) فجوة المهارات: اذكر المهارات والأدوات التقنية الناقصة التي يطلبها
   أصحاب العمل ولم يذكرها المستخدم.
3) خطة التعلم: قدم خطوات تعلم مرتبة حسب الأولوية بناءً على مدى تكرار
   المهارة في سوق العمل.
4) المصادر والمراجع: اذكر أسماء الشركات أو المسميات الوظيفية المحددة
   المستخرجة من السياق أعلاه كدليل على كل توصية.
""".strip()
    return prompt


def call_openrouter(prompt: str, api_key: str = None, model: str = None) -> str:
    """Send the prompt to the configured OpenRouter model and return the
    text of the response."""
    api_key = api_key or OPENROUTER_API_KEY
    model = model or OPENROUTER_MODEL

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Provide it via environment "
            "variables or Streamlit secrets -- never hard-code it."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate_roadmap(current_skills: str, target_job_title: str, retrieved_context: list,
                      api_key: str = None, model: str = None) -> str:
    """End-to-end helper: build the prompt and call the LLM."""
    prompt = build_prompt(current_skills, target_job_title, retrieved_context)
    return call_openrouter(prompt, api_key=api_key, model=model)


def main():
    # Manual smoke test using stage 6's retrieval.
    from importlib import import_module
    _retrieve = import_module("06_retrieve_context")

    current_skills = "Python, Excel, SQL"
    target_job_title = "Data Analyst"
    hits = _retrieve.retrieve_context(current_skills, target_job_title, top_k=8)

    report = generate_roadmap(current_skills, target_job_title, hits)
    print(report)


if __name__ == "__main__":
    main()
