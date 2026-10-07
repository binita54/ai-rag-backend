# Palm Mind AI RAG Backend

Conversational RAG backend assessment for the Palm Mind AI/ML Intern to AI Engineer role.

A FastAPI backend planned to provide two REST APIs: document ingestion (PDF/TXT text
extraction, selectable chunking strategies, embeddings, storage in Qdrant, metadata in
SQLite) and conversational RAG (multi-turn chat with Redis memory, a custom RAG pipeline
without prebuilt chains, LLM-generated answers, and LLM-driven interview booking).

## Planned technologies

- FastAPI with Pydantic Settings
- Qdrant (vector database)
- Redis (chat memory)
- SQLite via SQLAlchemy async (metadata and bookings)
- sentence-transformers (all-MiniLM-L6-v2 embeddings)
- OpenAI-compatible LLM provider, configurable via environment variables
- pytest and Ruff

## Setup (planned)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit the values
uvicorn app.main:app --reload
```

Note: on Windows, installing the CPU-only PyTorch build first avoids the large CUDA
download: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Current status

Project scaffolding and configuration only (Step 2 of the implementation plan). The
`/health` endpoint is live. Document ingestion, embeddings, Qdrant integration, Redis
memory, RAG, chat, and booking functionality are not implemented yet.
