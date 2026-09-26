from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

from evaluation import run_evaluation
from generate import generate_clinical_response
from ingest import contains_identifier, load_corpus
from models import QueryRequest, QueryResponse
from retrieve import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = load_corpus()
    app.state.corpus = corpus
    app.state.retriever = HybridRetriever(corpus["chunks"], corpus["documents"])
    yield


app = FastAPI(
    title="Northstar knowledge desk",
    description="Local, passage-cited retrieval over a synthetic healthcare operations corpus.",
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
    )


@app.post("/api/v1/evaluate")
async def evaluate(request: Request) -> dict[str, Any]:
    return run_evaluation(request.app.state.retriever)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
