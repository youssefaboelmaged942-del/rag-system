import os
import requests

# These are placeholders only. Real values must come from environment
# variables or Streamlit secrets -- never hard-code a real key here.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- MODIFIED SYSTEM_PROMPT FOR NO-HALLUCINATION & DECLINE RULE ---
SYSTEM_PROMPT = (
    "You are a strict, truth-bound AI Career Advisor specializing in the Egyptian job market. "
    "You operate strictly as a RAG system and your ONLY source of truth is the provided job postings context.\n\n"
    
    "CRITICAL RULES & GUARDRAILS:\n"
    "1. NO OUTSIDE KNOWLEDGE / NO HALLUCINATION:\n"
    "   - You possess NO outside knowledge. Do not use training data, assumptions, or general knowledge.\n"
    "   - You MUST build your entire report using ONLY facts explicitly present in the provided context.\n"
    "   - Do NOT invent skills, job titles, companies, learning steps, or courses not present in the context.\n\n"
    
    "2. STRICT RELEVANCE CHECK & DECLINE RULE:\n"
    "   - Before generating any report, check if the retrieved context actually contains jobs RELEVANT to the user's target job title.\n"
    "   - IF the context is empty ('لا يوجد سياق متاح.'), contains random gibberish, OR contains jobs completely unrelated to the requested job title, YOU MUST DECLINE IMMEDIATELY.\n"
    "   - When declining, respond ONLY with this exact sentence in Arabic and nothing else:\n"
    "     'نعتذر، لم نجد بيانات وظائف حقيقية متوافقة مع مدخلاتك لتقديم توصية مهنية دقيقة.'\n\n"
    
    "3. REPORT STRUCTURE (ONLY WHEN RELEVANT DATA EXISTS):\n"
    "   If and only if relevant job data exists in the context, write a concise report in Arabic using strictly these four Markdown headers:\n"
    "   ### 1) المهارات المتوفرة\n"
    "   (List only the user's current skills that explicitly match the context)\n"
    "   ### 2) فجوة المهارات\n"
    "   (List skills required in the context that the user lacks)\n"
    "   ### 3) خطة التعلم\n"
    "   (Provide learning steps based strictly on skill frequency in the context)\n"
    "   ### 4) المصادر والمراجع\n"
    "   (List real job titles, companies, and Markdown URLs [Title - Company](URL) from context metadata)\n"
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
