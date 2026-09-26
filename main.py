from fastapi import FastAPI, HTTPException
from models import QueryRequest, QueryResponse, SourceDocument
from database import get_db_pool
from retrieve import hybrid_search
from generate import generate_clinical_response
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="Zero-Trust Clinical RAG System",
    description="A highly elegant and robust RAG implementation prioritizing clinical safety.",
    version="1.0.0"
)

# Global database pool reference
db_pool: asyncpg.Pool = None

@app.on_event("startup")
async def startup_event():
    """Initialize DB connection on startup."""
    global db_pool
    try:
        db_pool = await get_db_pool()
        print("Database connected.")
    except Exception as e:
        print(f"Warning: Could not connect to database on startup: {e}")
        # Note: In production, we'd want this to fail loudly, 
        # but for local hackathon demo, we allow it to start and fail on request.

@app.on_event("shutdown")
async def shutdown_event():
    """Close DB connection on shutdown."""
    if db_pool:
        await db_pool.close()

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_clinical_knowledge(request: QueryRequest):
    """
    Main endpoint for zero-trust clinical retrieval and generation.
    Enforces RBAC, performs hybrid search, and safely generates responses.
    """
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection not available.")
        
    try:
        # 1. Retrieve & Filter (Hybrid Search + RBAC)
        top_docs = await hybrid_search(
            pool=db_pool, 
            query=request.query, 
            user_role=request.user_role,
            top_k=5
        )
        
        # 2. Generate (Safety Prompting)
        answer = await generate_clinical_response(
            query=request.query, 
            retrieved_docs=top_docs
        )
        
        # 3. Format citations and sources
        sources = [
            SourceDocument(
                id=doc['id'],
                content=doc['content'],
                metadata=doc['metadata'],
                score=doc['fusion_score']
            ) for doc in top_docs
        ]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")

if __name__ == "__main__":
    import uvicorn
    # Make sure to set the OPENAI_API_KEY environment variable before running
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
