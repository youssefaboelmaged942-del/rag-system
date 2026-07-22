import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Accept either filename so the pipeline works with the instructions'
# expected name or the name of the file actually supplied.
WUZZUF_JOBS_CANDIDATES = ["Wuzzuf_Jobs_2.csv", "Wuzzuf_Jobs.csv"]
DATA_ANALYST_JOBS_CANDIDATES = ["DataAnalystJobs.csv"]

UNIFIED_COLUMNS = [
    "Title", "Company", "Location", "Type", "Level",
    "YearsExp", "Country", "Skills", "SourceURL",
]

RAW_CACHE_PATH = os.path.join(DATA_DIR, "01_raw.pkl")

_TYPE_KEYWORDS = [
    "Full Time", "Part Time", "Freelance / Project",
    "Work From Home", "Internship", "Shift Based",
]


def _find_file(candidates, data_dir=DATA_DIR):
    for name in candidates:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_csv_path(data_dir: str = DATA_DIR) -> str:
    """Kept for backwards compatibility: resolve the primary Wuzzuf CSV."""
    path = _find_file(WUZZUF_JOBS_CANDIDATES, data_dir)
    if path is None:
        raise FileNotFoundError(
            f"Could not find a Wuzzuf jobs CSV in {data_dir}. "
            f"Expected one of: {WUZZUF_JOBS_CANDIDATES}"
        )
    return path


def _load_wuzzuf_jobs(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the original Wuzzuf_Jobs.csv (already close to the unified schema)."""
    path = _find_file(WUZZUF_JOBS_CANDIDATES, data_dir)
    if path is None:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.read_csv(path)
    required = ["Title", "Company", "Location", "Type", "Level", "YearsExp", "Country", "Skills"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df = df[required].copy()
    df["SourceURL"] = ""  # not available for this dataset
    return df[UNIFIED_COLUMNS]


def _split_type(type_str: str) -> str:
    """Insert readable separators into concatenated Type values like
    'Full TimeWork From Home' -> 'Full Time, Work From Home'."""
    if not isinstance(type_str, str):
        return type_str
    remaining = type_str
    found = []
    changed = True
    while changed:
        changed = False
        for kw in _TYPE_KEYWORDS:
            if remaining.startswith(kw):
                found.append(kw)
                remaining = remaining[len(kw):]
                changed = True
                break
    if not found:
        return type_str.strip()
    return ", ".join(found)


def _split_location(location_str: str):
    """Split 'District, Governorate, Country' or 'District, Country' into
    (location, country) so it lines up with how the first dataset uses
    Location (city/district) + Country (governorate-like field)."""
    if not isinstance(location_str, str):
        return location_str, ""
    parts = [p.strip() for p in location_str.split(",") if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[-2]  # district, governorate (drop trailing "Egypt")
    if len(parts) == 2:
        return parts[0], parts[-1]
    if len(parts) == 1:
        return parts[0], ""
    return "", ""


_EXP_RANGE_RE = re.compile(r"(\d+)\s*(?:\+|-\s*(\d+))?")


def _years_exp_bucket(years_exp: str, job_type: str) -> str:
    """Approximate a 'Level' bucket from years-of-experience + type, since
    DataAnalystJobs.csv doesn't include an explicit Level column.
    This is a DERIVED / approximate field, not authentic posting data --
    documented clearly in the README and explanation PDF."""
    if isinstance(job_type, str) and "Internship" in job_type:
        return "Student"
    if isinstance(job_type, str) and "Freelance / Project" in job_type and "Full Time" not in job_type:
        return "Freelance / Project"

    if not isinstance(years_exp, str):
        return "Not Specified"

    match = _EXP_RANGE_RE.search(years_exp)
    if not match:
        return "Not Specified"

    low = int(match.group(1))
    high = int(match.group(2)) if match.group(2) else low

    if high <= 2:
        return "Entry Level"
    if high <= 5:
        return "Experienced"
    return "Senior Management"


def _load_data_analyst_jobs(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load DataAnalystJobs.csv and map it onto the unified schema."""
    path = _find_file(DATA_ANALYST_JOBS_CANDIDATES, data_dir)
    if path is None:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.read_csv(path)
    required = ["JobTitle", "CompanyName", "Location", "Type", "Skills", "Experience", "Link"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    out = pd.DataFrame()
    out["Title"] = df["JobTitle"]
    out["Company"] = df["CompanyName"]

    split_loc = df["Location"].apply(_split_location)
    out["Location"] = split_loc.apply(lambda t: t[0])
    out["Country"] = split_loc.apply(lambda t: t[1])

    out["Type"] = df["Type"].apply(_split_type)
    out["YearsExp"] = df["Experience"].astype(str).str.strip()
    out["Level"] = [
        _years_exp_bucket(exp, typ) for exp, typ in zip(df["Experience"], out["Type"])
    ]
    out["Skills"] = df["Skills"].astype(str).str.strip()
    out["SourceURL"] = df["Link"].astype(str).str.strip()

    return out[UNIFIED_COLUMNS]


def load_documents(csv_path: str = None, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and merge all available Wuzzuf job sources into one dataframe
    with the unified schema.

    Parameters
    ----------
    csv_path : str, optional
        Deprecated single-file override, kept for backwards compatibility.
        If given, only that file is loaded (assumed to already match the
        original Wuzzuf_Jobs.csv schema).
    data_dir : str, optional
        Folder to search for known source files.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        df["SourceURL"] = df.get("SourceURL", "")
        return df[UNIFIED_COLUMNS]

    wuzzuf_df = _load_wuzzuf_jobs(data_dir)
    data_analyst_df = _load_data_analyst_jobs(data_dir)

    combined = pd.concat([wuzzuf_df, data_analyst_df], ignore_index=True)
    if combined.empty:
        raise FileNotFoundError(
            f"No known Wuzzuf datasets found in {data_dir}. Expected one of "
            f"{WUZZUF_JOBS_CANDIDATES} and/or {DATA_ANALYST_JOBS_CANDIDATES}."
        )
    return combined


def main():
    df = load_documents()
    print(f"Loaded {len(df)} job postings (merged sources) with columns: {list(df.columns)}")
    with_url = (df["SourceURL"].astype(str).str.len() > 0).sum()
    print(f"Rows with a real source URL: {with_url}")
    print(df.head(3))
    print(df.tail(3))

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_pickle(RAW_CACHE_PATH)
    print(f"Saved raw dataframe cache to {RAW_CACHE_PATH}")


if __name__ == "__main__":
    main()
