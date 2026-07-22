"""
05_create_chroma_store.py
--------------------------
Stage 5 of the RAG pipeline: Vector Storage & Indexing.

Initializes a local, persistent ChromaDB collection and stores the
chunk embeddings from 04_vector_representation.py together with their
text and metadata, so 06_retrieve_context.py can query them later.

Run directly:
    python 05_create_chroma_store.py
to (re)build the Chroma collection on disk under chroma_store/.
"""

import os
import numpy as np

from importlib import import_module

_vec = import_module("04_vector_representation")
DATA_DIR = _vec.DATA_DIR
EMBEDDINGS_PATH = _vec.EMBEDDINGS_PATH
CHUNKS_PATH = _vec.CHUNKS_PATH
load_chunks = _vec.load_chunks

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "wuzzuf_jobs"


def get_chroma_collection(persist_directory: str = CHROMA_DIR):
    """Return the (created if needed) persistent Chroma collection."""
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, collection


def build_store():
    if not os.path.exists(EMBEDDINGS_PATH):
        _vec.main()

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch between number of chunks ({len(chunks)}) and "
            f"embeddings ({embeddings.shape[0]}); re-run stage 04."
        )

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client, collection = get_chroma_collection()

    # Chroma collections are append-only for a given id, so clear any
    # previous run's data before re-ingesting a fresh build.
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Chroma expects plain Python lists of floats.
    embeddings_list = embeddings.tolist()

    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings_list[start:end],
        )

    return collection


def main():
    collection = build_store()
    print(f"Chroma collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    print(f"Persisted to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
