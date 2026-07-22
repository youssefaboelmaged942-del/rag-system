"""
06_retrieve_context.py
-----------------------
Stage 6 of the RAG pipeline: Context Retrieval Engine.

Accepts user query parameters ('Current Skills' and 'Target Job
Title'), embeds the target job title, and performs semantic vector
search over ChromaDB combined with a metadata filter on job title to
retrieve the top relevant Wuzzuf job postings.

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

    Combines:
      - Semantic vector search over the full collection using an
        embedding built from the target job title + current skills
        (so both influence relevance ranking).
      - A metadata filter that prioritizes postings whose title
        contains the target job title, when available, falling back
        to an unfiltered semantic search otherwise.

    Returns a list of dicts: {"text": str, "metadata": dict, "distance": float}
    """
    _, collection = get_chroma_collection(CHROMA_DIR)

    query_text = f"Target Job Title: {target_job_title}. Current Skills: {current_skills}"
    query_embedding = embed_texts([query_text])[0].tolist()

    # First try a metadata-filtered search restricted to job titles that
    # contain the target title (case-insensitive "contains" match).
    where_document = {"$contains": target_job_title} if target_job_title else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where_document=where_document,
        )
    except Exception:
        results = None

    filtered_hits = _format_results(results)

    if len(filtered_hits) >= min(3, top_k):
        return filtered_hits

    # Fall back to a pure semantic search across the whole collection
    # if the metadata filter was too strict (e.g. no exact title match).
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return _format_results(results)


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
