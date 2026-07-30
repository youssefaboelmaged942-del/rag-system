import os
import json
import pandas as pd

from importlib import import_module

_pre = import_module("02_preprocessing")
DATA_DIR = _pre.DATA_DIR
CLEAN_CACHE_PATH = _pre.CLEAN_CACHE_PATH
CHUNKS_PATH = os.path.join(DATA_DIR, "03_chunks.jsonl")


def build_chunk_text(row: pd.Series) -> str:
    text = (
        f"Job Title: {row['Title']} | "
        f"Level: {row['Level']} | "
        f"Experience: {row['YearsExp']} | "
        f"Location: {row['Location']}, {row['Country']} | "
        f"Company: {row['Company']} | "
        f"Required Skills: {row['Skills']}"
    )
    source_url = row.get("SourceURL", "")
    if isinstance(source_url, str) and source_url:
        text += f" | Source: {source_url}"
    return text


def build_chunk_metadata(row: pd.Series) -> dict:
    return {
        "title": row["Title"],
        "level": row["Level"],
        "location": row["Location"],
        "source_url": row.get("SourceURL", "") or "",
    }


def chunk_documents(df: pd.DataFrame) -> list:
    """Turn each row of the cleaned dataframe into a chunk dict:
    {"id": str, "text": str, "metadata": dict}
    """
    chunks = []
    for i, row in df.iterrows():
        chunk = {
            "id": f"job_{i}",
            "text": build_chunk_text(row),
            "metadata": build_chunk_metadata(row),
        }
        chunks.append(chunk)
    return chunks


def main():
    if not os.path.exists(CLEAN_CACHE_PATH):
        _pre.main()
    cleaned_df = pd.read_pickle(CLEAN_CACHE_PATH)

    chunks = chunk_documents(cleaned_df)
    print(f"Built {len(chunks)} chunks. Example:")
    print(chunks[0])

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
