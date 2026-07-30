from importlib import import_module

_store = import_module("05_create_chroma_store")
get_chroma_collection = _store.get_chroma_collection
CHROMA_DIR = _store.CHROMA_DIR

_vec = import_module("04_vector_representation")
embed_texts = _vec.embed_texts

DEFAULT_DISTANCE_THRESHOLD = None 


def retrieve_context(
    current_skills: str, 
    target_job_title: str, 
    top_k: int = 8, 
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> list:
    """Retrieve the top_k most relevant job postings for a user query.

    Uses semantic vector search over the full collection using an
    embedding built from the target job title + current skills.

    Returns a list of dicts: {"text": str, "metadata": dict, "distance": float}
    Filters out any results that exceed the distance_threshold.
    """
    _, collection = get_chroma_collection(CHROMA_DIR)

    query_text = f"Target Job Title: {target_job_title}. Current Skills: {current_skills}"
    query_embedding = embed_texts([query_text])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # إرسال النتائج مع حد المسافة للفلترة
    formatted_results = _format_results(results, distance_threshold=distance_threshold)

    print(f"DEBUG: Retrieved {len(formatted_results)} relevant job postings for query: '{target_job_title}'")

    return formatted_results


def _format_results(results, distance_threshold: float = None) -> list:
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[None] * len(docs)])[0]

    formatted = []
    for doc, meta, dist in zip(docs, metas, distances):
        # الفلترة: إذا كان هنالك مسافة محددة وتجاوزت الحد المسموح، يتم استبعاد النتيجة
        if dist is not None and distance_threshold is not None:
            if dist > distance_threshold:
                continue
        
        formatted.append({"text": doc, "metadata": meta, "distance": dist})

    return formatted


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
