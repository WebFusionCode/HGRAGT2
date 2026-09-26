from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The clinical question to answer.")
    user_role: str = Field(
        default="clinician", 
        description="RBAC role of the user requesting information (e.g., 'clinician', 'patient')."
    )

class SourceDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float = Field(..., description="The RRF fusion score or retrieval score.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer with inline citations or epistemic refusal.")
    sources: List[SourceDocument] = Field(..., description="List of documents retrieved and used as context.")
