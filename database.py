import os
import asyncpg
from pgvector.asyncpg import register_vector
from dotenv import load_dotenv

load_dotenv()

# We default to a standard local postgres url, update as needed
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/clinical_rag")

async def get_db_pool() -> asyncpg.Pool:
    """
    Creates and returns a connection pool to the PostgreSQL database.
    Registers the pgvector extension on the connection.
    """
    # Note: Ensure the database exists. 
    # For a real setup, handle connection errors gracefully.
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    # We must register the vector type with asyncpg for it to parse vectors correctly
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await register_vector(conn)
        
    return pool

async def init_db(pool: asyncpg.Pool):
    """
    Initializes the database schema for the clinical chunks.
    Sets up pgvector and full-text search capabilities.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            -- Table for storing chunked clinical documents
            DROP TABLE IF EXISTS clinical_chunks CASCADE;
            CREATE TABLE IF NOT EXISTS clinical_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                embedding vector(384), -- Dimension for all-MiniLM-L6-v2
                chunk_metadata JSONB NOT NULL,
                -- Generated column for PostgreSQL Full-Text Search (Sparse Search)
                search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            );
            
            -- HNSW index for fast approximate nearest neighbor (ANN) vector search
            CREATE INDEX IF NOT EXISTS clinical_chunks_embedding_idx 
            ON clinical_chunks USING hnsw (embedding vector_cosine_ops);
            
            -- GIN index for fast sparse keyword search (BM25 alternative)
            CREATE INDEX IF NOT EXISTS clinical_chunks_fts_idx
            ON clinical_chunks USING GIN (search_vector);
            
            -- GIN index on metadata for fast RBAC filtering
            CREATE INDEX IF NOT EXISTS clinical_chunks_metadata_idx
            ON clinical_chunks USING GIN (chunk_metadata);
        """)
