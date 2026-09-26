# VoughtCrop

A local-first healthcare knowledge retrieval demo. It indexes fictional organization documents and, by default, fetches a small allowlisted set of public clinical pages at API startup. It filters passages by role before ranking, quotes retrieved evidence with passage citations, surfaces curated organization-policy conflicts, and refuses questions the accessible corpus cannot support.

All organization records and policies are synthetic. Public clinical pages are fetched live and may be unavailable or change without notice. The demo does not call a hosted answer model and is not for patient care.

## Run locally

Use Python 3.10 or newer. Install the dependencies once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API and Streamlit interface in separate terminals:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`. API documentation is available at `http://localhost:8000/docs`.

## Included corpus

`data/synthetic/corpus_manifest.json` describes each source's owner, version, dates, status, conflict topic, and allowed roles. The corpus includes a sectioned guideline, operational SOPs, a superseded memo, a row-heavy formulary CSV, a payer policy, a device manual, structured inventory JSON, and patient education.

The ingestion path supports Markdown, text, CSV, JSON, XLSX, PDF, and DOCX. Install dependencies from `requirements.txt` for the optional office and PDF parsers. Markdown headings are kept as section metadata; long sections are split into overlapping passages; CSV and workbook rows and JSON records remain individually citable. Stable passage IDs are derived from source, section, and passage content.

Add a source file and manifest entry, then restart the API to rebuild the in-memory index. A source can list `allowed_roles`, `status` (`active`, `superseded`, or `retired`), `effective_date`, `review_date`, and optional `conflict_keys` plus a curated `stance`. Conflict handling compares curated keys and stances; it does not claim to discover every contradiction automatically.

## Public guideline ingestion

At API startup, `main.py` scrapes and validates the allowlisted CDC, AAFP, NCBI/PMC, and Cleveland Clinic URLs in `scrape_ingest.py`, then adds usable pages directly to the in-memory retriever used by chat. Pages under 200 characters or containing common block-page markers are skipped and logged; there is no fabricated fallback in the live chat index. Disable live fetching with `ENABLE_PUBLIC_WEB_SOURCES=0`. The API does not read from PostgreSQL.

`scrape_ingest.py` also provides a separate optional PostgreSQL/pgvector persistence path. Configure `DATABASE_URL` and LiteLLM provider credentials in the environment or an untracked `.env`, then run:

```bash
python scrape_ingest.py --dry-run
python scrape_ingest.py
```

Scrape validation rejects short pages and common access-block messages, and prints a short extraction preview. If every URL fails, one clinician-scoped fallback passage is tagged `demo_only`, `unverified`, and without a citation URL; never treat it as clinical guidance. To clear the database table before a demo, run `python clear_db.py --confirm`; this irreversibly truncates `clinical_chunks`.

The default embedding model is `openai/text-embedding-3-small` with 384 dimensions. Set `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` together when using another model; the script validates the database vector dimension before writing. A run fetches and embeds all successfully extracted pages before transactionally replacing those URLs, preserving the previous indexed passages if embedding or insertion fails. Source roles are stored with each passage; any future database-backed retriever must enforce `allowed_roles` in its SQL query.

## Retrieval and answer behavior

The default mode combines BM25 lexical retrieval, role-scoped TF-IDF cosine similarity, reciprocal-rank fusion, and a freshness-aware rerank. The TF-IDF representation is the local fallback and is not a neural embedding model. To enable local sentence-transformer vectors, install `sentence-transformers`, set `ENABLE_NEURAL_EMBEDDINGS=1`, and restart the API. The selected model runs locally; the first model download may require network access.

Responses are extractive: each displayed claim is copied from its cited passage, and citation expanders open the exact source text and its section, row, or record locator. A query is refused when evidence is weak, a specific requested term is missing, or only retired/superseded evidence is available. Review-due and superseded sources are labeled. Conflicts are returned with both passages and version dates.

Common identifier patterns are redacted during ingestion. Queries matching direct-identifier patterns are blocked before retrieval and are not echoed into chat history. This is a narrow safety control, not a complete de-identification system; the demo corpus is synthetic and must stay that way.

## Access model

The API filters each user's accessible chunks before computing BM25 or vector scores. The Streamlit role selector demonstrates those filters and clears the visible chat when the role changes. The role is explicitly simulated and can be forged by a caller; a production deployment must derive it from verified identity and policy claims, and enforce organization-specific authorization in the service.

## Evaluation

Run the included eight-question benchmark:

```bash
python3 evaluate.py
```

The report checks expected-source recall, refusal accuracy, curated conflict detection, claim-to-source citation coverage, exact passage support, ACL leaks, and identifier leakage. The Quality view runs the same authored benchmark through the API's currently loaded corpus. Public-source availability can change the retrieval results; scores are not a clinical safety certification or a generalization estimate.

## Operational limits

- The local index is rebuilt from the manifest and live public pages at API startup; there is no multi-tenant persistence, audit log, document upload flow, or production identity provider. PostgreSQL ingestion is separate and is not queried by this demo API.
- Roles and conflict keys require governance. Do not treat UI role simulation as authentication.
- Direct-identifier patterns do not detect every name or indirect identifier. Never enter real patient information.
- The default vector fallback is TF-IDF. Enable and evaluate a neural encoder against the target corpus before relying on it.
- All policies, formularies, dates, devices, and organization names in the demo are invented.
