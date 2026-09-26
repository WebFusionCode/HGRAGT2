import json
from typing import List, Dict, Any
import asyncpg
from sentence_transformers import SentenceTransformer

# Load model once at module level for reuse
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def reciprocal_rank_fusion(dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
    """
    Combines dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
    RRF Score = sum(1 / (k + rank)) for each rank in the result lists.
    """
    rrf_scores: Dict[str, float] = {}
    docs_map: Dict[str, Dict[str, Any]] = {}
    
    # Process dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc['id']
        docs_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Process sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc['id']
        docs_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort documents by their RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return sorted documents with their new fusion scores
    fused_results = []
    for doc_id, score in sorted_docs:
        doc_data = docs_map[doc_id]
        doc_data['fusion_score'] = score
        fused_results.append(doc_data)
        
    return fused_results

async def hybrid_search(pool: asyncpg.Pool, query: str, user_role: str, top_k: int = 5) -> List[Dict]:
    """
    Performs a hybrid search using both vector similarity (pgvector) and 
    PostgreSQL full-text search (BM25 approx). 
    Enforces Zero-Trust by pre-filtering chunks based on RBAC user_role.
    """
    try:
        # 1. Get query embedding via local model
        query_embedding = embed_model.encode(query).tolist()
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return []
    
    async with pool.acquire() as conn:
        # 2. Dense Vector Search (Cosine Similarity <=> operator)
        # Pre-filter by JSONB metadata 'allowed_role'
        dense_query = """
            SELECT id, content, chunk_metadata, 
                   1 - (embedding <=> $1) AS similarity_score
            FROM clinical_chunks
            WHERE chunk_metadata->>'allowed_role' = $2
            ORDER BY embedding <=> $1
            LIMIT 10;
        """
        dense_records = await conn.fetch(dense_query, query_embedding, user_role)
        dense_results = [
            {
                "id": str(r["id"]), 
                "content": r["content"], 
                "metadata": json.loads(r["chunk_metadata"]), 
                "score": r["similarity_score"]
            }
            for r in dense_records
        ]
        
        # 3. Sparse Keyword Search (PostgreSQL Full-Text Search)
        # Pre-filter by JSONB metadata 'allowed_role'
        sparse_query = """
            SELECT id, content, chunk_metadata, 
                   ts_rank(search_vector, plainto_tsquery('english', $1)) AS rank_score
            FROM clinical_chunks
            WHERE chunk_metadata->>'allowed_role' = $2
              AND search_vector @@ plainto_tsquery('english', $1)
            ORDER BY rank_score DESC
            LIMIT 10;
        """
        sparse_records = await conn.fetch(sparse_query, query, user_role)
        sparse_results = [
            {
                "id": str(r["id"]), 
                "content": r["content"], 
                "metadata": json.loads(r["chunk_metadata"]), 
                "score": r["rank_score"]
            }
            for r in sparse_records
        ]
        
    # 4. Fuse Results using RRF
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
    
    # Return top K results
    return fused_results[:top_k]
