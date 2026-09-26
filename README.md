# Northstar knowledge desk

A local-first healthcare knowledge retrieval demo. It indexes fictional organization documents, filters passages by a selected role before ranking, quotes retrieved evidence with passage citations, surfaces curated version conflicts, and refuses questions the accessible corpus cannot support.

All included records and organization policies are synthetic. The demo does not call a hosted answer model and is not for patient care.

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

`scrape_ingest.py` is a separate PostgreSQL/pgvector ingestion path for the configured CDC and AAFP public pages. Install the added dependencies, configure `DATABASE_URL` and LiteLLM provider credentials in the environment or an untracked `.env`, then run:

```bash
python scrape_ingest.py --dry-run
python scrape_ingest.py
```

The default embedding model is `openai/text-embedding-3-small` with 384 dimensions. Set `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` together when using another model; the script validates the database vector dimension before writing. A run fetches and embeds all successfully extracted pages before transactionally replacing those URLs, preserving the previous indexed passages if embedding or insertion fails. Source roles are stored with each passage (`patient`/`clinician` for CDC HEADS UP, `clinician` for HCP sources); downstream retrieval must enforce `allowed_roles` in its database query. The current demo API still reads only the local in-memory synthetic corpus and does not query this PostgreSQL table.

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

The report checks expected-source recall, refusal accuracy, curated conflict detection, claim-to-source citation coverage, exact passage support, ACL leaks, and identifier leakage. The Quality view runs the same benchmark through the API. Scores are for this authored synthetic set only; they are not a clinical safety certification or a generalization estimate.

## Operational limits

- The index is rebuilt in memory from the local manifest at API startup; there is no multi-tenant persistence, audit log, document upload flow, or production identity provider.
- Roles and conflict keys require governance. Do not treat UI role simulation as authentication.
- Direct-identifier patterns do not detect every name or indirect identifier. Never enter real patient information.
- The default vector fallback is TF-IDF. Enable and evaluate a neural encoder against the target corpus before relying on it.
- All policies, formularies, dates, devices, and organization names in the demo are invented.
