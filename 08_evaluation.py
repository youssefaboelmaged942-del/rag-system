import os
from importlib import import_module

_retrieve = import_module("06_retrieve_context")
_prompting = import_module("07_prompting")

retrieve_context = _retrieve.retrieve_context
generate_roadmap = _prompting.generate_roadmap

REQUIRED_SECTIONS = [
    "المهارات المتوفرة",
    "فجوة المهارات",
    "خطة التعلم",
    "المصادر والمراجع",
]

TOP_K = 8

# Ground truth: for each query, a set of (Title, Company) pairs manually
# verified to exist in data/Wuzzuf_Jobs.csv / data/DataAnalystJobs.csv
# after cleaning (see 02_preprocessing.py's Title.str.title() casing).
# A test case "hits" if the retrieved top-K includes ANY one of these.
TEST_CASES = [
    {
        "current_skills": "Python, SQL, Excel, Power BI",
        "target_job_title": "Data Analyst",
        "ground_truth": [
            ("Data Analyst", "United Distributors"),
            ("Data Analyst", "Itcan"),
            ("Data Analyst", "Safa International Travel"),
            ("Data Analyst", "London International Patient Services"),
            ("Marketing Data Analyst", "Nile Projects & Trading"),
        ],
    },
    {
        "current_skills": "Recruitment, Communication, Microsoft Office",
        "target_job_title": "HR Specialist",
        "ground_truth": [
            ("Hr Specialist", "Unival"),
            ("Hr & Admin Officer", "Maktech 3D"),
            ("Hr Payroll Specialist", "Marbella For Food Industry"),
        ],
    },
    {
        "current_skills": "Sales, Negotiation, CRM, English",
        "target_job_title": "Sales Representative",
        "ground_truth": [
            ("Outdoor Sales Representative", "Confidential"),
            ("Sales Executive - Damietta", "Hamza Group"),
            ("Sales Engineer - Cairo", "Hamza Group"),
            ("Sales Supervisor - Tanta", "Oppo Egypt"),
        ],
    },
    {
        "current_skills": "Accounting, Excel, Financial Reporting",
        "target_job_title": "Accountant",
        "ground_truth": [
            ("Accountant", "Unimix Egypt For Readymix Concrete"),
            ("Cost Accountant", "Chema Foam"),
            ("Warehouse Accountant", "Confidential"),
            ("Accounting Manager", "Wnpro"),
        ],
    },
    {
        "current_skills": "Customer Service, Call Center, English",
        "target_job_title": "Customer Service Agent",
        "ground_truth": [
            ("Customer Service Agent", "Johnson Controls"),
            ("Customer Service Representative", "Electronic House"),
            ("Customer Service Agent - Irish Account", "Majorel Egypt"),
            ("Customer Service Agent Vodafone Uk/Ie - Giza", "Majorel Egypt"),
        ],
    },
]


def _extract_title_company(hit: dict):
    """Pull (title, company) out of one retrieved hit, case-insensitively."""
    title = str(hit.get("metadata", {}).get("title", "")).strip().lower()
    company = ""
    text = hit.get("text", "")
    if "Company: " in text:
        company = text.split("Company: ")[1].split(" | ")[0].strip().lower()
    return title, company


def evaluate_retrieval(case: dict) -> dict:
    """Run retrieval for one test case and check it against ground truth."""
    hits = retrieve_context(
        current_skills=case["current_skills"],
        target_job_title=case["target_job_title"],
        top_k=TOP_K,
    )

    retrieved_pairs = {_extract_title_company(h) for h in hits}
    ground_truth_pairs = {(t.lower(), c.lower()) for t, c in case["ground_truth"]}

    found = ground_truth_pairs & retrieved_pairs
    recall_at_k = len(found) / len(ground_truth_pairs) if ground_truth_pairs else 0.0
    hit_at_k = len(found) > 0  # at least one ground-truth posting was retrieved

    return {
        "hits": hits,
        "found": found,
        "recall_at_k": recall_at_k,
        "passed": hit_at_k,
    }


def evaluate_answer(case: dict, hits: list) -> dict:
    """Generate a report for one test case and check structure + grounding."""
    report = generate_roadmap(
        current_skills=case["current_skills"],
        target_job_title=case["target_job_title"],
        retrieved_context=hits,
    )

    missing_sections = [s for s in REQUIRED_SECTIONS if s not in report]
    has_all_sections = not missing_sections

    # Groundedness: does the report mention at least one company/title that
    # genuinely appears in the retrieved context? Catches the LLM ignoring
    # the context and inventing an answer.
    context_terms = set()
    for hit in hits:
        title, company = _extract_title_company(hit)
        if title:
            context_terms.add(title)
        if company:
            context_terms.add(company)

    report_lower = report.lower()
    grounded = any(term in report_lower for term in context_terms if len(term) > 3)

    return {
        "report": report,
        "has_all_sections": has_all_sections,
        "missing_sections": missing_sections,
        "grounded": grounded,
        "passed": has_all_sections and grounded,
    }


def main():
    print("=" * 70)
    print("تقييم الاسترجاع (Retrieval Evaluation) -- مقارنة بـ Ground Truth حقيقي")
    print("=" * 70)

    retrieval_results = []
    for case in TEST_CASES:
        result = evaluate_retrieval(case)
        retrieval_results.append((case, result))
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(
            f"{status} | {case['target_job_title']:<25} | "
            f"recall@{TOP_K}={result['recall_at_k']:.2f} "
            f"({len(result['found'])}/{len(case['ground_truth'])} من الوظائف المتوقعة ظهرت)"
        )
        if not result["passed"]:
            print(f"        الوظائف المتوقعة (ولم تظهر): {case['ground_truth']}")

    retrieval_pass_rate = sum(r["passed"] for _, r in retrieval_results) / len(retrieval_results)
    print(f"\nنسبة نجاح الاسترجاع (Hit@{TOP_K}): {retrieval_pass_rate * 100:.0f}% "
          f"({sum(r['passed'] for _, r in retrieval_results)}/{len(retrieval_results)})")

    print("\n" + "=" * 70)
    print("تقييم الإجابة (Answer Evaluation)")
    print("=" * 70)

    if not (os.environ.get("OPENROUTER_API_KEY") or _prompting.OPENROUTER_API_KEY):
        print(
            "⚠️  تم تخطي تقييم الإجابة: OPENROUTER_API_KEY غير موجود في "
            "متغيرات البيئة. عرّفه وشغّل الملف تاني عشان يتم تقييم جودة "
            "التقارير المولّدة فعلياً."
        )
        return

    answer_results = []
    for case, retrieval_result in retrieval_results:
        hits = retrieval_result["hits"]
        if not hits:
            print(f"⏭️  تخطي '{case['target_job_title']}' -- لا يوجد سياق مسترجع لتوليد تقرير منه.")
            continue
        try:
            result = evaluate_answer(case, hits)
        except Exception as e:
            print(f"❌ FAIL | {case['target_job_title']:<25} | خطأ أثناء التوليد: {e}")
            continue

        answer_results.append((case, result))
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        detail = []
        if result["missing_sections"]:
            detail.append(f"أقسام ناقصة: {', '.join(result['missing_sections'])}")
        if not result["grounded"]:
            detail.append("لم يُعثر على أي إشارة لشركة/وظيفة حقيقية من السياق")
        detail_str = f" ({'; '.join(detail)})" if detail else ""
        print(f"{status} | {case['target_job_title']:<25}{detail_str}")

    if answer_results:
        answer_pass_rate = sum(r["passed"] for _, r in answer_results) / len(answer_results)
        print(f"\nنسبة نجاح الإجابة: {answer_pass_rate * 100:.0f}% "
              f"({sum(r['passed'] for _, r in answer_results)}/{len(answer_results)})")


if __name__ == "__main__":
    main()
