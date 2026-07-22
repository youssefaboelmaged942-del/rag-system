"""
04_vector_representation.py
----------------------------
Stage 4 of the RAG pipeline: Vector Representation & Embeddings.

Converts the structured textual chunks from 03_chunking.py into dense
vector representations using a high-performing sentence embedding
model (sentence-transformers). A multilingual model is used because
users may describe their current skills in Arabic or English while
the Wuzzuf job postings are in English.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Good quality/speed trade-off, runs locally (no API key needed),
    and supports both Arabic and English text.

Run directly:
    python 04_vector_representation.py
to embed all chunks and save the embeddings for
05_create_chroma_store.py.
"""

import os
import json
import numpy as np

from importlib import import_module

_chunking = import_module("03_chunking")
DATA_DIR = _chunking.DATA_DIR
CHUNKS_PATH = _chunking.CHUNKS_PATH
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "04_embeddings.npy")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None  # lazy-loaded singleton so importers (06/07) don't reload it


def get_embedding_model():
    """Load (once) and return the sentence-transformers embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_chunks(path: str = CHUNKS_PATH) -> list:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def embed_texts(texts: list) -> np.ndarray:
    """Embed a list of raw strings (used both for chunks and user queries)."""
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def main():
    if not os.path.exists(CHUNKS_PATH):
        _chunking.main()
    chunks = load_chunks()

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME} ...")
    embeddings = embed_texts(texts)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
