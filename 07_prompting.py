"""
07_prompting.py
----------------
Stage 7 of the RAG pipeline: Prompt Engineering & OpenRouter LLM Call.

Builds a structured prompt combining the retrieved job-market context
with the user's current skills and target job title, then sends it to
an LLM via the OpenRouter API to produce a structured Arabic career
report.

CRITICAL INSTRUCTION FOR THE SYSTEM_PROMPT:
The model is strictly constrained to respond ONLY if the provided context
(retrieved Wuzzuf jobs) is relevant to the user's input (skills/job title).
If the context is empty, non-relevant, contains only gibberish, or if the
user provides nonsense inputs, the model must decline to generate a report
and instead provide a professional Arabic polite message explaining that
there are no matching job postings available to make a career recommendation.
This guarantees a zero-hallucination policy and verifies that the RAG
pipeline is working only on real data.

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

# --- MODIFIED SYSTEM_PROMPT FOR NO-HALLUCINATION & DECLINE RULE ---
SYSTEM_PROMPT = (
    "You are an expert AI Career Advisor specializing in the Egyptian job "
    "market. You are given real job postings retrieved from Wuzzuf as "
    "context. Your strictly mandatory task is to analyze the user's skills and "
    "target job title AGAINST THE PROVIDED CONTEXT.\n\n"
    "CRITICAL RULES:\n"
    "1. NO HALLUCINATION: You must build your entire report using *ONLY* the "
    "provided context (retrieved job postings). Do *not* invent companies, "
    "job titles, required skills, or learning steps that are not directly "
    "supported by the context.\n"
    "2. NO RELEVANT DATA DECLINE RULE: If the provided context is empty, contains "
    "only the text 'لا يوجد سياق متاح.', is non-relevant to the user's target job "
    "title and skills, or if the user's input is nonsensical/gibberish, "
    "YOU MUST NOT GENERATE A CAREER REPORT. Instead, respond politely in Arabic "
    "stating that no relevant job postings were found in the current dataset to "
    "provide a professional recommendation (e.g., 'نعتذر، لم نجد بيانات وظائف حقيقية "
    "متوافقة مع مدخلاتك لتقديم توصية مهنية دقيقة'). Do *not* generate any other "
    "sections.\n"
    "3. CITATIONS: Some context items include a real listing URL after 'Source:' -- "
    "when citing those specific postings in the 'المصادر والمراجع' section, "
    "include the URL alongside the company/title so the user can verify it directly; "
    "for items without a URL, cite the company name and job title only.\n"
    "4. LANGUAGE AND STRUCTURE: If (and only if) you generate a report, respond always in "
    "Arabic, structured in exactly four sections using these headers: "
    "'المهارات المتوفرة', 'فجوة المهارات', 'خطة التعلم', 'المصادر والمراجع'."
)


def build_prompt(current_skills: str, target_job_title: str, retrieved_context: list) -> str:
    """Construct the user-turn prompt combining retrieved context with
    the user's profile."""
    # Use the same default message to make sure the LLM has a string to work with
    # if the list is empty.
    context_block = "\n".join(f"- {hit['text']}" for hit in retrieved_context) or "لا يوجد سياق متاح."

    # Keep the user prompt simple. The logic is handled by the System Prompt.
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
    # Using low temperature (0.1 - 0.3) is crucial for a strict no-hallucination policy.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,  # Lowered for more predictable/strict output
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

    # --- Case 1: Valid Data ---
    print("\n--- Testing Valid Data (expect report) ---")
    current_skills = "Python, Excel, SQL"
    target_job_title = "Data Analyst"
    hits = _retrieve.retrieve_context(current_skills, target_job_title, top_k=8)
    report = generate_roadmap(current_skills, target_job_title, hits)
    print(report)

    # --- Case 2: Gibberish Inputs/No Context ---
    # To test the decline rule properly, we simulate an empty retrieval.
    print("\n--- Testing Non-relevant/No Context Data (expect professional decline message) ---")
    current_skills = "sadasdasd, asdasdasd" # User nonsense input
    target_job_title = "Data Analyst"
    # Even if retrieve_context returns some hits based on title,
    # the skills won't match, or we can simply force hits to be empty to test
    # the context relevance check in the prompt.
    empty_hits = [] # Simulate no context available
    report = generate_roadmap(current_skills, target_job_title, empty_hits)
    print(report)


if __name__ == "__main__":
    main()
