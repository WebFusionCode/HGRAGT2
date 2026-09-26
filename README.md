# VoughtCrop

VoughtCrop is a demonstration clinical knowledge desk for asking questions across healthcare guidance and operational documents. It brings together a small, fictional hospital corpus and a curated set of public clinical pages, then returns short answers that can be checked against the passages they came from.

The system is deliberately cautious: when the material it can access does not support an answer, it says what is missing instead of filling the gap with a guess. When registered documents disagree, it shows the competing passages and their dates.

This is a hackathon prototype, not a clinical decision-support product. All hospital names, staff-facing policies, formularies, payer rules, and records in the included corpus are fictional. Never enter real patient information or use this tool to make a care decision.

## What Happens to a Question

The default chat path is local retrieval, not a general-purpose chatbot. It does not send questions to an answer-generating LLM.

```text
Synthetic files                         Allowlisted public pages
      |                                           |
      +---- parse, validate, redact, and chunk ---+
                          |
                 in-memory source index
                          |
Question -> identifier check -> role filter -> BM25 + TF-IDF retrieval
                          |                -> reciprocal-rank fusion
                          |                -> freshness reranking
                          v
             extractive claims + passage citations
                          |
            answer, conflict, or stated refusal

Optional clinician-only path, after local refusal or review-due evidence:
Question -> de-identified general terms -> NCBI PubMed abstracts -> citations
```

In practical terms, the pipeline works like this:

1. The API loads the source manifest and files under `data/synthetic/`. On startup, it can also fetch pages from the allowlist in `scrape_ingest.py`.
2. Ingestion extracts document sections, table rows, or structured records, checks public page text, redacts common identifier patterns, and divides long text into citable passages.
3. Each passage keeps source details such as its document, section or row, version, dates, status, and permitted roles.
4. The retriever removes passages the selected role is not allowed to see before calculating relevance. It combines lexical and local TF-IDF rankings, then reranks using reciprocal rank and freshness signals.
5. The answer layer quotes claims from retrieved passages and attaches passage citations. It can refuse when evidence is weak or too specific to the question, or report a curated conflict when documents with the same conflict key take different positions.
6. If a clinician explicitly opts in to PubMed fallback, and local evidence is refused or marked review-due, the service may search a limited set of PubMed abstracts. That search is separate from the organization's policies and is labeled accordingly.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`
- Internet access for startup retrieval of public pages and, when enabled, PubMed fallback

Create an environment and install the project dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the Application

Run the API and the Streamlit interface in two terminal windows. Activate `.venv` in both.

Terminal 1, start the API:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The API builds the in-memory corpus during startup. Public websites can respond slowly or block automated requests, so the number of loaded public sources may vary. The terminal logs which pages were accepted or skipped.

Terminal 2, start the interface:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open these local addresses:

- Application: <http://127.0.0.1:8501>
- API health: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>

The app reads its API address from `KNOWLEDGE_API_URL`; it defaults to `http://127.0.0.1:8000`. If port 8000 is already occupied, start the API on another port, for example `8001`, and set `KNOWLEDGE_API_URL=http://127.0.0.1:8001` before starting Streamlit.

To run without fetching public pages at API startup, set `ENABLE_PUBLIC_WEB_SOURCES=0` in the API terminal before launching it. The synthetic corpus remains available.

## Using the Knowledge Desk

Choose a demo role in the interface, enter a question, and inspect the returned citations. The available roles are Clinician, Operations, Billing administrator, and Patient. The role changes which passages are eligible for retrieval; it is not a login or identity check.

Try questions that exercise different source types:

- “What should staff check before an assisted transfer?”
- “Can staff reuse a transfer sling with visible contamination after wiping it?”
- “Does Medication Beta require prior authorization?”
- “Which mobile lifts are available and where are they located?”
- “What initial rest and return-to-activity advice does the concussion guidance give?”
- “What dose of amoxicillin is approved for a child?”

The first questions should find relevant evidence in the demo corpus. The sling question demonstrates a conflict between a current procedure and an older memo. The pediatric dose question is intentionally unsupported and should be refused.

Open a citation to see the exact passage, its section, table row, or record locator, and a link to the public page when one exists. The Corpus view shows sources available to the selected role. The Quality view runs the authored benchmark against the API's currently loaded corpus.

### PubMed Fallback

The `PubMed fallback` checkbox is an explicit opt-in and is off by default. It is available only to the Clinician demo role. If enabled, the API calls NCBI only when local evidence is refused or a retrieved source is marked review-due; it does not search PubMed for every question. A locally detected conflict is shown instead of being bypassed with an external search.

The fallback searches for selected guideline and review publication types and retrieves at most six abstracts. It does not retrieve full text, write results to the local index, or establish organizational policy. Abstracts are labeled as external research evidence. If you enable this option, general query terms are sent to NCBI. Direct identifier patterns are blocked, but the detector is not a complete privacy filter. Do not include patient names, dates of birth, record numbers, or other patient details.

The fallback intentionally does not run for organization-specific policy, formulary, coverage, approval, operational, or dosage questions. Those need authoritative, applicable local evidence. The application does not scrape Google results or use Wikipedia as clinical evidence.

## Sources and Ingestion

### Included synthetic corpus

`data/synthetic/corpus_manifest.json` describes the included sources and their metadata. The sample set includes:

- A sectioned clinical guideline and patient education document
- Operational procedures, including a superseded memo used to demonstrate disagreement
- A formulary table and a payer policy
- A device manual
- Structured equipment inventory records

The parser supports Markdown, plain text, CSV, JSON, XLSX, PDF, and DOCX. Markdown headings are retained as section context. Long sections are split into overlapping passages; table rows and structured records remain separately identifiable for citation.

To add an internal source, place the synthetic or approved de-identified file under `data/synthetic/`, add its entry to `corpus_manifest.json`, and restart the API. Set its `allowed_roles`, `status`, `effective_date`, and `review_date` carefully. For a known disagreement, assign matching `conflict_keys` to the documents and give each a meaningful `stance`. Conflict detection depends on this curated metadata; it is not an automatic fact checker that discovers every contradiction.

### Public page fetching

The live API uses only the URLs listed in `scrape_ingest.py`. The scraper checks the host allowlist and `robots.txt`, uses Trafilatura to extract the main page text, and skips short or obvious block/error pages. A page that cannot be fetched or extracted is logged and omitted; the live API does not turn a failed scrape into made-up evidence.

The public source set may change as websites change their pages, access rules, or robots policy. Check `/health` and the API startup log to see what loaded for the current run. You can add a target URL in `TARGET_URLS` only when it is an appropriate authoritative source and its host is permitted by the scraper's allowlist.

### Optional PostgreSQL and pgvector ingestion

The standalone `scrape_ingest.py` script has a separate persistence path for PostgreSQL/pgvector. This is not the storage backend used by the default chat API: the API builds and searches its in-memory index at startup.

For a validation-only run that does not write to the database:

```bash
python scrape_ingest.py --dry-run
```

For a real ingestion run, set `DATABASE_URL` and the embedding provider credentials required by your LiteLLM configuration, then run:

```bash
python scrape_ingest.py
```

The default embedding model is `openai/text-embedding-3-small`, configured for 384 dimensions. If you use a different model, set `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` to match the model and existing database table. The script validates dimensions and replaces indexed passages for fetched URLs transactionally. Keep credentials in environment variables or a local, untracked `.env` file; never commit secrets.

The destructive database cleanup command is:

```bash
python clear_db.py --confirm
```

It truncates the `clinical_chunks` table. Use it only when you intentionally want to remove its contents and have verified the configured database.

## API

The API is useful for testing retrieval separately from the interface.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness, loaded source counts, retrieval mode, and ingestion redaction count |
| `GET` | `/api/v1/catalog?user_role=clinician` | List sources visible to a demo role |
| `POST` | `/api/v1/query` | Retrieve evidence and return an answer, citations, conflict, or refusal |
| `POST` | `/api/v1/evaluate` | Run the authored evaluation against the API's current index |

Example local query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What should staff check before an assisted transfer?","user_role":"clinician","include_public_search":false}'
```

Set `include_public_search` to `true` only when you intentionally want to allow the clinician-only PubMed fallback described above.

## Evaluation

Run the deterministic benchmark from the repository root:

```bash
python -m unittest discover -s tests -v
python evaluate.py
```

The small authored question set checks expected-source retrieval, citation and passage support, refusal behavior, curated conflict detection, role-access leaks, and identifier leakage. It includes answerable questions as well as unsupported and identifier-containing cases. The API's Quality view evaluates the sources loaded in that running instance, which may include public pages; `python evaluate.py` uses the local synthetic corpus. Results can therefore differ.

This is a useful regression check, not a broad clinical benchmark, external validation, or safety certification. A perfect score on eight authored questions would not establish clinical reliability.

## Safety and Current Limits

- All organization-specific demo content is fictional. Keep real patient information out of questions, files, logs, and evaluation data.
- Identifier handling uses pattern-based checks. It can miss names and indirect identifiers, so it is not a de-identification guarantee.
- Demo roles are caller-selected and can be forged. A real deployment needs verified identity, authorization policy, tenant isolation, and audit logging. The current in-memory retriever does filter role-inaccessible passages before scoring, but the UI selector is not authentication.
- The normal answer path quotes retrieved text; it does not prove that the source itself is correct, current, or clinically appropriate for an individual patient.
- The default local similarity signal is TF-IDF, not a neural embedding. To try a local Sentence Transformer, install `sentence-transformers`, set `ENABLE_NEURAL_EMBEDDINGS=1`, and restart the API. `SENTENCE_TRANSFORMER_MODEL` selects the model; the default is `all-MiniLM-L6-v2`. Evaluate it against the target corpus before relying on it.
- Conflict handling only catches disagreements represented by curated conflict keys and stances. Dates and status labels help review sources but do not replace an organization's document-governance process.
- The live index is rebuilt at API startup and held in memory. There is no production document upload workflow, multi-tenant persistence, verified identity provider, or production audit trail in this prototype.
- PostgreSQL ingestion and the live API index are separate paths. Loading passages into pgvector does not make them appear in chat.

## Troubleshooting

**The interface says the API is unavailable.** Confirm the API terminal is still running and that `KNOWLEDGE_API_URL` points to the same host and port. Open `/health` directly to check readiness.

**The API reports that port 8000 is already in use.** Start it on another port, such as 8001, and set `KNOWLEDGE_API_URL=http://127.0.0.1:8001` before starting Streamlit.

**A public-source question returns insufficient evidence.** Check the API startup log and `/health`; the relevant website may have blocked or changed its page. The scraper skips unusable text rather than indexing an error page. Try a more specific question that matches the available passages, or add and review an authoritative source in the corpus. The optional PubMed fallback is limited and is not a replacement for missing organizational policy.

**A database ingestion run fails.** Confirm `DATABASE_URL`, provider credentials, table availability, and the configured vector dimensions. Use `--dry-run` first to inspect extraction without writing. Do not run the cleanup command as a generic troubleshooting step; it deletes the table contents.
