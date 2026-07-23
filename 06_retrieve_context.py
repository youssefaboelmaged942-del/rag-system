"""
06_retrieve_context.py
-----------------------
Stage 6 of the RAG pipeline: Context Retrieval Engine.

Accepts user query parameters ('Current Skills' and 'Target Job
Title'), embeds the target job title, and performs semantic vector
search over ChromaDB to retrieve the top relevant Wuzzuf job postings.

This module is imported by 07_prompting.py and by streamlit_app.py --
it is not meant to rebuild the store on import, only to query it.
"""

from importlib import import_module

_store = import_module("05_create_chroma_store")
get_chroma_collection = _store.get_chroma_collection
CHROMA_DIR = _store.CHROMA_DIR

_vec = import_module("04_vector_representation")
embed_texts = _vec.embed_texts


def retrieve_context(current_skills: str, target_job_title: str, top_k: int = 8) -> list:
    """Retrieve the top_k most relevant job postings for a user query.

    Uses semantic vector search over the full collection using an
    embedding built from the target job title + current skills.

    Returns a list of dicts: {"text": str, "metadata": dict, "distance": float}
    """
    _, collection = get_chroma_collection(CHROMA_DIR)

    query_text = f"Target Job Title: {target_job_title}. Current Skills: {current_skills}"
    query_embedding = embed_texts([query_text])[0].tolist()

    # المباشرة بالبحث الدلالي لضمان إرجاع نتائج صحيحة دائماً وتفادي مشاكل الفلاتر الصارمة
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted_results = _format_results(results)

    # طباعة بسيطة للـ Debugging لمعرفة عدد الوظائف التي تم استرجاعها
    print(f"DEBUG: Retrieved {len(formatted_results)} job postings for query: '{target_job_title}'")

    return formatted_results


def _format_results(results) -> list:
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[None] * len(docs)])[0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def main():
    # Simple manual smoke test.
    hits = retrieve_context(
        current_skills="Python, Excel, SQL",
        target_job_title="Data Analyst",
        top_k=5,
    )
    for h in hits:
        print(f"- {h['text']}")


if __name__ == "__main__":
    main()
