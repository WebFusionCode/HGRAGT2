import asyncio
import json
import os
from database import get_db_pool, init_db
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# We no longer need an OpenAI API key for embeddings, 
# as we are running all-MiniLM-L6-v2 locally.


# Minimal synthetic clinical dataset for MVP
MOCK_CLINICAL_DATA = [
    {
        "text": "Concussion Management: Initial rest for 24-48 hours is recommended. After this period, patients should begin a gradual return to activity, staying below their symptom-exacerbation threshold. Avoid NSAIDs in the first 48 hours due to the risk of intracranial bleeding.",
        "allowed_role": "clinician",
        "doc_id": "guideline-concussion-001"
    },
    {
        "text": "Achilles Tendinopathy: First-line treatment involves load management and a progressive loading program, such as the Alfredson protocol (eccentric exercises) or heavy slow resistance training. Corticosteroid injections are generally not recommended due to increased risk of tendon rupture.",
        "allowed_role": "clinician",
        "doc_id": "guideline-achilles-001"
    },
    {
        "text": "Patient advice for Achilles pain: Rest the tendon, apply ice for 15-20 minutes, and wear supportive shoes. Do not try to stretch the tendon excessively if it's acutely painful.",
        "allowed_role": "patient",
        "doc_id": "patient-edu-achilles-001"
    },
    {
        "text": "Contradicting Guideline (Old): For Achilles Tendinopathy, immediate corticosteroid injections are highly recommended to reduce inflammation rapidly.",
        "allowed_role": "clinician",
        "doc_id": "guideline-achilles-OLD-002"
    }
]

async def embed_and_store():
    """
    Chunks the mock dataset, generates local embeddings, 
    and inserts them into the pgvector database with RBAC metadata.
    """
    pool = await get_db_pool()
    await init_db(pool)
    
    print("Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Ingesting mock clinical documents...")
    async with pool.acquire() as conn:
        for item in MOCK_CLINICAL_DATA:
            print(f"Generating embedding for {item['doc_id']}...")
            try:
                # Generate embedding using local SentenceTransformers model
                embedding = model.encode(item["text"]).tolist()
                
                metadata = {
                    "allowed_role": item["allowed_role"],
                    "doc_id": item["doc_id"]
                }
                
                # Insert into DB
                await conn.execute("""
                    INSERT INTO clinical_chunks (content, embedding, chunk_metadata)
                    VALUES ($1, $2, $3)
                """, item["text"], embedding, json.dumps(metadata))
                
                print(f"✅ Successfully inserted chunk for {item['doc_id']}")
                
            except Exception as e:
                print(f"❌ Failed to process {item['doc_id']}: {e}")
                
    await pool.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    asyncio.run(embed_and_store())
