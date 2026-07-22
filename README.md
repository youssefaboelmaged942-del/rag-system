# AI Career Advisor (Wuzzuf RAG Project)

A retrieval-augmented generation (RAG) assistant that compares a user's
**current skills** against real job postings scraped from **Wuzzuf**
to find the target **job title**'s skill gap, and produces a
structured Arabic career roadmap with citations back to real job
listings.

## Pipeline

```
01_documents.py            -> load raw Wuzzuf CSV
02_preprocessing.py        -> clean nulls/duplicates/special chars, standardize casing
03_chunking.py              -> one structured text chunk + metadata per job posting
04_vector_representation.py -> embed chunks (sentence-transformers, multilingual)
05_create_chroma_store.py   -> persist embeddings + metadata in a local ChromaDB store
06_retrieve_context.py      -> semantic search + metadata filter on target job title
07_prompting.py              -> build prompt, call OpenRouter LLM, Arabic structured report
streamlit_app.py             -> user-facing UI, wires stages 06 + 07 together
```

## Data

Two Wuzzuf sources are merged by `01_documents.py` into one unified
schema: `Title, Company, Location, Type, Level, YearsExp, Country,
Skills, SourceURL`.

1. `data/Wuzzuf_Jobs.csv` — 4,380 general job postings, columns already
   close to the unified schema. Has no listing URL, so `SourceURL` is
   left empty for these rows.
2. `data/DataAnalystJobs.csv` — 1,695 additional postings focused on
   data/analyst roles, with a different raw schema (`JobTitle,
   CompanyName, Location, Date, Link, Type, Skills, Experience`).
   `01_documents.py` maps it onto the unified schema:
   - `Location` (e.g. "Zamalek, Cairo, Egypt") is split into a city-level
     `Location` and a governorate-level `Country`, mirroring how the
     first dataset uses those two fields.
   - `Type` values that were concatenated without a separator (e.g.
     `"Full TimeWork From Home"`) are split back into readable,
     comma-separated values.
   - `Level` is **derived** (not present in the raw file) from years of
     experience and job type — e.g. 0-2 yrs → "Entry Level", 3-5 yrs →
     "Experienced", 6+ yrs → "Senior Management", internships →
     "Student". This is an approximation, not authentic posting data,
     and is documented here and in the explanation PDF for transparency.
   - `Link` is kept as `SourceURL` — a real, verifiable job-posting URL.
     `03_chunking.py` appends it to the chunk text and metadata, and
     `07_prompting.py` instructs the LLM to cite it directly in the
     "المصادر والمراجع" section whenever it's available, which makes
     those citations stronger than a company/title pair alone.
   - `Date` (a relative string like "1 day ago") is dropped — not a
     real timestamp, no analytical value here.
   - Six rows had a leftover HTML entity (`&amp;`) in `Skills` from the
     scrape; `02_preprocessing.py` now runs `html.unescape()` before
     stripping characters, so e.g. "Health &amp; Safety" becomes
     "Health & Safety" correctly instead of losing the `;`.

After merging and cleaning (deduping, whitespace/special-char
stripping, casing standardization), the pipeline runs on **6,071**
job postings total, of which **1,695** carry a real source URL.

> Note: the project instructions reference a file named
> `Wuzzuf_Jobs_2.csv` for the first source. `01_documents.py` looks for
> either name, so the pipeline works with the dataset as supplied.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline once, stage by stage, to build the local vector store:

```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py
```

Each stage caches its output under `data/` (and `chroma_store/` for
stage 5), so later stages don't need to be re-run unless the data or
chunking logic changes.

## API key (local development)

Copy `.env.example` to `.env` and fill in your real key — **never**
commit `.env` or upload it in the submission ZIP:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Then export it before running Streamlit locally, e.g.:

```bash
export OPENROUTER_API_KEY=sk-or-...
streamlit run streamlit_app.py
```

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (make sure `.env` and `chroma_store/` are
   **not** included — see `.gitignore`).
2. Because Streamlit Cloud gives you a fresh filesystem, run stages
   01-05 as a one-time setup step (e.g. a small build script or a
   `st.cache_resource` bootstrap at the top of `streamlit_app.py`) so
   the Chroma store exists before the app serves requests.
3. In the app: **Manage app -> Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. Deploy. `streamlit_app.py` reads the key from `st.secrets` at
   runtime, falling back to environment variables if secrets aren't
   configured (useful for local testing).

## Final submission checklist

- [x] All required Python files exist (01-07, streamlit_app.py).
- [x] `requirements.txt` exists.
- [ ] Real API key is not included in the ZIP or GitHub repo (double-check
      before you submit — `.env` is gitignored but verify manually).
- [ ] Streamlit secrets configured in valid TOML on the deployed app.
- [ ] The Streamlit app runs successfully end-to-end.
- [x] The answer uses retrieved context (see `06_retrieve_context.py`).
- [x] The answer cites sources (see "المصادر والمراجع" section of the
      generated report, and the expandable "الوظائف المسترجعة" panel
      in the UI).
