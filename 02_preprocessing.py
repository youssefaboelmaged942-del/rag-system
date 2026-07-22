"""
02_preprocessing.py
--------------------
Stage 2 of the RAG pipeline: Data Preprocessing & Cleaning.

Cleans the merged job postings dataframe produced by 01_documents.py:
- Drops rows with missing/null values in the required text fields
  (SourceURL is exempt -- it's legitimately empty for the
  Wuzzuf_Jobs.csv source, which has no listing URLs).
- Drops duplicate job entries.
- Removes special, non-informative characters from text fields
  (SourceURL is exempt -- stripping "special characters" would break
  real URLs like https://wuzzuf.net/jobs/p/...?o=1&l=sp).
- Standardizes text casing (title case for job titles/companies,
  consistent casing for level and skills) for reliable querying later.
- Splits Skills entries that use either comma- or double-space-
  separated formatting (the two source datasets use different
  conventions) into a single, consistent comma-separated format.

Run directly:
    python 02_preprocessing.py
to clean the cached raw dataframe and save the result for
03_chunking.py.
"""

import os
import re
import html
import pandas as pd

from importlib import import_module

_docs = import_module("01_documents")
DATA_DIR = _docs.DATA_DIR
RAW_CACHE_PATH = _docs.RAW_CACHE_PATH
CLEAN_CACHE_PATH = os.path.join(DATA_DIR, "02_cleaned.pkl")

# SourceURL is intentionally excluded: it may be empty (Wuzzuf_Jobs.csv has
# no listing URLs) and must not be special-char-stripped (it's a real URL).
REQUIRED_NONEMPTY_COLUMNS = ["Title", "Company", "Location", "Type", "Level", "YearsExp", "Country", "Skills"]
# Skills is handled separately by _clean_skills(), which must run on the RAW
# text so it can still see whichever separator (comma or double-space) the
# source dataset used -- the generic whitespace collapsing below would
# otherwise destroy the double-space separator before it can be split on.
TEXT_COLUMNS = ["Title", "Company", "Location", "Type", "Level", "Country"]

# Characters that add noise but no meaning (keep letters, numbers, spaces,
# commas, slashes and hyphens since skills/titles legitimately use those).
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s,/\-&]", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_DOUBLE_SPACE_SPLIT_RE = re.compile(r"\s{2,}")


def _clean_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = html.unescape(value)
    value = value.strip()
    value = _SPECIAL_CHARS_RE.sub("", value)
    value = _MULTI_SPACE_RE.sub(" ", value).strip()
    return value


def _clean_skills(skills_str: str) -> str:
    """Normalize skills into a single comma-separated string regardless of
    which source dataset it came from: Wuzzuf_Jobs.csv already uses
    commas; DataAnalystJobs.csv uses double-spaces as a separator.
    Runs on the RAW value (before generic whitespace collapsing) so the
    double-space separator is still intact when we split on it."""
    if not isinstance(skills_str, str):
        return skills_str
    skills_str = html.unescape(skills_str)
    if "," in skills_str:
        raw_parts = skills_str.split(",")
    else:
        raw_parts = _DOUBLE_SPACE_SPLIT_RE.split(skills_str)
    parts = [_clean_text(p) for p in raw_parts]
    parts = [p for p in parts if p]
    return ", ".join(parts)


def clean_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the merged job postings dataframe.

    Returns a new dataframe with nulls/duplicates removed and text
    fields cleaned + consistently cased.
    """
    df = df.copy()

    # 1. Remove missing/null values in required fields (SourceURL exempt).
    df = df.dropna(subset=REQUIRED_NONEMPTY_COLUMNS)
    df = df[(df[REQUIRED_NONEMPTY_COLUMNS].astype(str) != "").all(axis=1)]

    # 2. Strip special characters and extra whitespace (text columns only).
    for col in TEXT_COLUMNS:
        df[col] = df[col].apply(_clean_text)

    # SourceURL: only trim whitespace, never strip URL-special characters.
    if "SourceURL" in df.columns:
        df["SourceURL"] = df["SourceURL"].apply(lambda v: v.strip() if isinstance(v, str) else v)

    # 3. Standardize casing.
    df["Title"] = df["Title"].str.title()
    df["Company"] = df["Company"].str.title()
    df["Location"] = df["Location"].str.title()
    df["Country"] = df["Country"].str.title()
    df["Level"] = df["Level"].str.title()
    df["YearsExp"] = df["YearsExp"].str.replace(" Yrs Of Exp", "", regex=False, case=False)
    df["YearsExp"] = df["YearsExp"].astype(str).str.strip()

    df["Skills"] = df["Skills"].apply(_clean_skills)

    # 4. Drop duplicate job entries (same title/company/location/skills).
    df = df.drop_duplicates(subset=["Title", "Company", "Location", "Skills"])
    df = df.reset_index(drop=True)

    return df


def main():
    if not os.path.exists(RAW_CACHE_PATH):
        _docs.main()  # produce the raw cache if it doesn't exist yet
    raw_df = pd.read_pickle(RAW_CACHE_PATH)

    cleaned_df = clean_documents(raw_df)
    print(f"Rows before cleaning: {len(raw_df)} -> after cleaning: {len(cleaned_df)}")
    print(cleaned_df.head(3))
    print(cleaned_df.tail(3))

    cleaned_df.to_pickle(CLEAN_CACHE_PATH)
    print(f"Saved cleaned dataframe cache to {CLEAN_CACHE_PATH}")


if __name__ == "__main__":
    main()
