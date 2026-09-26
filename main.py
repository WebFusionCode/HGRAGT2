from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import logging
import os
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from evaluation import run_evaluation
from generate import generate_clinical_response
from ingest import contains_identifier, load_corpus, redact_identifiers
from models import QueryRequest, QueryResponse
from pubmed_search import search_pubmed_abstracts
from retrieve import HybridRetriever

_LOGGER = logging.getLogger("northstar.api")


def _load_public_guidelines(corpus: dict[str, Any]) -> None:
    if os.getenv("ENABLE_PUBLIC_WEB_SOURCES", "1").strip().lower() in {"0", "false", "no"}:
        corpus["web_document_count"] = 0
        return

    try:
        from scrape_ingest import TARGET_URLS, make_session, make_chunk_records, scrape_url, validate_extracted_text
    except ImportError as exc:
        _LOGGER.warning("Public guidelines are disabled because scraper dependencies are unavailable (%s).", type(exc).__name__)
        corpus["web_document_count"] = 0
        return

    session = make_session()
    robots_cache: dict[str, Any] = {}
    web_documents: list[dict[str, Any]] = []
    web_chunks: list[dict[str, Any]] = []
    web_redactions = 0
    try:
        for url in TARGET_URLS:
            try:
                document = scrape_url(session, url, robots_cache)
                valid, reason = validate_extracted_text(document.text)
                if not valid:
                    _LOGGER.warning("Skipping public source %s: %s", url, reason)
                    continue

                safe_text, redaction_count = redact_identifiers(document.text)
                document.text = safe_text
                document.content_sha256 = hashlib.sha256(safe_text.encode("utf-8")).hexdigest()
                records = make_chunk_records(document)
                if not records:
                    _LOGGER.warning("Skipping public source %s: no passages were generated", url)
                    continue

                doc_id = "WEB-" + hashlib.sha256(document.source_url.encode("utf-8")).hexdigest()[:10].upper()
                allowed_roles = records[0]["metadata"]["allowed_roles"]
                source_path = document.source_url.lower()
                if "/heads-up/" in source_path:
                    source_type = "Patient concussion guidance"
                elif "/hcp/" in source_path:
                    source_type = "CDC clinical guideline resource"
                elif document.host.endswith("clevelandclinic.org"):
                    source_type = "Patient education"
                elif document.host.endswith(("ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov")):
                    source_type = "Clinical reference"
                else:
                    source_type = "AAFP clinical review"

                for record in records:
                    metadata = {**record["metadata"], "doc_id": doc_id}
                    web_chunks.append({
                        "id": str(record["id"]),
                        "doc_id": doc_id,
                        "content": record["content"],
                        "metadata": metadata,
                    })

                web_documents.append({
                    "doc_id": doc_id,
                    "title": document.title,
                    "source_type": source_type,
                    "publisher": document.host,
                    "version": document.published_date or document.content_sha256[:10],
                    "effective_date": document.published_date or "Not stated",
                    "review_date": "Not stated",
                    "status": "active",
                    "allowed_roles": allowed_roles,
                    "conflict_keys": [],
                    "stance": "",
                    "source_url": document.source_url,
                    "published_date": document.published_date,
                    "http_last_modified": document.http_last_modified,
                    "sha256": document.content_sha256,
                })
                web_redactions += redaction_count
                _LOGGER.info("Loaded public source %s (%s passages)", document.title, len(records))
            except Exception as exc:
                _LOGGER.warning("Could not load public source %s (%s)", url, type(exc).__name__)
    finally:
        session.close()

    corpus["documents"].extend(web_documents)
    corpus["chunks"].extend(web_chunks)
    corpus["redactions"] += web_redactions
    corpus["web_document_count"] = len(web_documents)
    corpus["web_redactions"] = web_redactions
    if web_documents:
        corpus["corpus_name"] += " + live public guidelines"
    else:
        _LOGGER.warning("No public guidelines loaded; the API is serving only its synthetic local corpus.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = load_corpus()
    _load_public_guidelines(corpus)
    app.state.corpus = corpus
    app.state.retriever = HybridRetriever(corpus["chunks"], corpus["documents"])
    yield


app = FastAPI(
    title="VoughtCrop",
    description="Passage-cited retrieval over synthetic organization records and validated public clinical sources.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    corpus = request.app.state.corpus
    retriever = request.app.state.retriever
    return {
        "status": "ready",
        "corpus_name": corpus["corpus_name"],
        "document_count": len(corpus["documents"]),
        "passage_count": len(corpus["chunks"]),
        "public_source_count": corpus.get("web_document_count", 0),
        "retrieval_mode": retriever.retrieval_mode,
        "identifier_redactions_at_ingestion": corpus["redactions"],
        "answer_mode": "extractive with source-level citation validation",
        "external_generation": False,
    }


@app.get("/api/v1/catalog")
async def catalog(request: Request, user_role: str = Query("clinician")) -> list[dict[str, Any]]:
    if user_role not in {"clinician", "operations", "billing_admin", "patient"}:
        raise HTTPException(status_code=422, detail="Unknown demo role.")
    return request.app.state.retriever.catalog(user_role)


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge(request: Request, payload: QueryRequest) -> QueryResponse:
    started = perf_counter()
    retriever: HybridRetriever = request.app.state.retriever

    if contains_identifier(payload.query):
        response = generate_clinical_response(payload.query, [])
        ranked_sources: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        retrieval_mode = retriever.retrieval_mode
        public_search_status = "identifier_blocked" if payload.include_public_search else "not_requested"
    else:
        result = retriever.search(payload.query, payload.user_role)
        ranked_sources = result["sources"]
        conflicts = result["conflicts"]
        retrieval_mode = result["retrieval_mode"]
        response = generate_clinical_response(
            payload.query,
            ranked_sources,
            conflicts=conflicts,
            missing_evidence=result["missing_evidence"],
            evidence_coverage=result["coverage"],
            matched_query_terms=result["matched_query_terms"],
            unmatched_specific_terms=result["unmatched_specific_terms"],
        )
        public_search_status = "not_requested"

        if payload.include_public_search:
            if payload.user_role != "clinician":
                public_search_status = "unsupported_role"
            elif response["decision"] == "conflict":
                public_search_status = "skipped_conflict"
            else:
                stale_local_evidence = any(source.get("freshness_status") == "review due" for source in ranked_sources)
                if response["decision"] != "refused" and not stale_local_evidence:
                    public_search_status = "not_needed"
                else:
                    try:
                        external_chunks, public_search_status = await run_in_threadpool(
                            search_pubmed_abstracts, payload.query, payload.user_role
                        )
                    except Exception as exc:
                        _LOGGER.warning("PubMed fallback unavailable (%s)", type(exc).__name__)
                        external_chunks = []
                        public_search_status = "unavailable"

                    if external_chunks:
                        external_retriever = HybridRetriever(external_chunks)
                        external_result = external_retriever.search(payload.query, payload.user_role)
                        external_response = generate_clinical_response(
                            payload.query,
                            external_result["sources"],
                            conflicts=external_result["conflicts"],
                            missing_evidence=external_result["missing_evidence"],
                            evidence_coverage=external_result["coverage"],
                            matched_query_terms=external_result["matched_query_terms"],
                            unmatched_specific_terms=external_result["unmatched_specific_terms"],
                        )
                        if external_response["decision"] == "answered":
                            response = external_response
                            ranked_sources = external_result["sources"]
                            conflicts = external_result["conflicts"]
                            retrieval_mode = f"{retriever.retrieval_mode} + PubMed fallback"
                            public_search_status = "answered"
                        else:
                            public_search_status = "no_support"
                            existing_ids = {source["id"] for source in ranked_sources}
                            ranked_sources.extend(
                                source for source in external_result["sources"]
                                if source["id"] not in existing_ids
                            )
                            response["missing_evidence"] = external_response["missing_evidence"] or response["missing_evidence"]

    elapsed_ms = (perf_counter() - started) * 1_000
    public_sources = [
        {**source, "title": source["metadata"].get("title", "Source passage")}
        for source in ranked_sources
    ]
    return QueryResponse(
        **response,
        sources=public_sources,
        conflicts=conflicts,
        retrieval_mode=retrieval_mode,
        processing_ms=round(elapsed_ms, 2),
        pii_blocked=response["decision"] == "identifier_blocked",
        corpus_redactions=request.app.state.corpus["redactions"],
        public_search_status=public_search_status,
    )


@app.post("/api/v1/evaluate")
async def evaluate(request: Request) -> dict[str, Any]:
    return run_evaluation(request.app.state.retriever)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
