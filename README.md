# 🏥 Zero-Trust Clinical RAG System

Welcome to your hackathon-ready Zero-Trust Clinical RAG System! The entire codebase has been successfully generated, installed, and configured in your current workspace: `/Users/harshsingh/Desktop/HGRAGT2`.

## 🏗️ Architecture & Features Built

This project strictly adheres to the requirements of prioritizing clinical safety, explicit epistemic refusals, and granular citations.

### 1. ⚙️ Tech Stack & Setup
* **Backend:** `FastAPI` implementation in `main.py` offering a lightweight, asynchronous `POST /query` endpoint.
* **Database:** `PostgreSQL` with `pgvector` enabled, configured via `database.py`. It uses a 384-dimensional vector space for local embeddings.
* **LLM Orchestration:** Text generation uses `litellm` in `generate.py` (currently configured to use your `Groq` API key with the lightning-fast `qwen/qwen3.8-27b` model).
* **Embeddings:** Fully local and free embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) to eliminate OpenAI API key issues.

### 2. 🧩 Core Modules

* **`ingest.py` (Structure & Access):** Parses mock clinical guidelines (Achilles tendinopathy, concussion management) into chunks and embeds them. Crucially, it attaches **RBAC metadata** (`{"allowed_role": "clinician"}` vs `"patient"`) to each chunk.
* **`retrieve.py` (Hybrid Search):** Implements a dual-retriever system. It performs a dense vector search using `pgvector` and a sparse keyword search (mocked BM25), combining them with **Reciprocal Rank Fusion (RRF)**. It strictly pre-filters the database based on the `user_role` argument.
* **`generate.py` (Safety & Refusal Prompting):** Uses a highly constrained system prompt enforcing:
  * (A) Exact chunk ID inline citations (e.g., `[guideline-concussion-001]`).
  * (B) Explicit statements of `"Insufficient evidence retrieved"` when context is lacking.
  * (C) Highlighting contradictions between retrieved documents.
* **`evaluate.py` (LLM-as-a-Judge):** A comprehensive evaluation script with 5 clinical QA pairs. It correctly penalizes hallucinations (scoring them `0`) and rewards fully grounded responses (scoring them `1`).

## 🚀 How to Run It

Since your terminal is defaulting to the Mac system Python, you must use your Miniforge environment to run the code.

**1. Start the API Server:**
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/uvicorn main:app --reload
```
*(Note: I already have this running in the background for you!)*

**2. Access the Interactive API Docs:**
Open your browser and navigate to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

**3. Run the Evaluation Script:**
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python3 evaluate.py
```

Good luck with the hackathon! Let me know if you want to tweak the safety prompts or add more mock data.
