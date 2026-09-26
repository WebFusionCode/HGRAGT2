from typing import Any, Literal

from pydantic import BaseModel, Field


UserRole = Literal["clinician", "operations", "billing_admin", "patient"]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1_000, description="A de-identified question about organizational knowledge.")
    user_role: UserRole = Field(default="clinician", description="Demo role used by the retrieval access filter.")


class AnswerClaim(BaseModel):
    text: str
    citations: list[str]
    source_status: str


class SourceDocument(BaseModel):
    id: str
    doc_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    score: float
    bm25_score: float
    vector_score: float
    freshness_status: str


class ConflictPassage(BaseModel):
    id: str
    doc_id: str
    title: str
    status: str
    effective_date: str | None = None
    stance: str
    content: str


class ConflictRecord(BaseModel):
    topic: str
    summary: str
    passages: list[ConflictPassage]


class QueryResponse(BaseModel):
    answer: str
    claims: list[AnswerClaim]
    sources: list[SourceDocument]
    conflicts: list[ConflictRecord]
    missing_evidence: list[str]
    decision: Literal["answered", "conflict", "refused", "identifier_blocked"]
    retrieval_mode: str
    processing_ms: float
    pii_blocked: bool = False
    corpus_redactions: int = 0
