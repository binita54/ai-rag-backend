# AI RAG Backend

A backend implementation of a document-grounded conversational RAG system
with interview booking, built with FastAPI, Qdrant, Redis, SQLAlchemy, and
an OpenAI-compatible LLM interface. It provides two REST APIs: document
ingestion and conversational RAG with multi-turn memory, plus an
interview-booking flow driven by an LLM tool call.

## Features

- PDF and TXT document ingestion
- Text extraction (PyMuPDF for PDF, UTF-8 for TXT)
- Two chunking strategies: recursive and sentence-aware
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`, normalized)
- Qdrant vector search (cosine similarity, local or remote mode)
- SQLite metadata persistence (documents and chunks)
- Custom RAG pipeline (explicitly orchestrated, no prebuilt chains)
- Redis conversation memory with history trimming and TTL
- Multi-turn conversations
- OpenAI-compatible LLM integration
- LLM function/tool calling
- Interview booking via LLM tool call (name, email, date, time)
- Pydantic validation of booking arguments
- Booking persistence in SQLite
- Duplicate-booking protection
- FastAPI REST APIs
- Automated tests (pytest) and Ruff linting

## Architecture

The application follows a ports-and-adapters layout: FastAPI routes validate
input and delegate to services, which depend on provider-agnostic ports and
are wired to concrete implementations through dependency injection.

Document ingestion:

```
Upload → Extract → Chunk → Persist metadata (SQLite) → Embed → Qdrant
```

Conversational RAG:

```
Question → Embed → Qdrant retrieval → Context construction → Redis history
→ LLM → Response + memory persistence
```

Interview booking:

```
Chat → LLM tool call → Pydantic validation → BookingService → SQLite
→ Confirmation → conversation memory
```

The RAG pipeline is implemented explicitly in `RAGService` (embedding,
retrieval, context construction, history replay, prompt assembly, generation,
memory persistence) rather than through a prebuilt RetrievalQA chain.

Layers: `api` (routes and dependency wiring), `schemas` (Pydantic models),
`services` (orchestration and providers), `ports` (interfaces), `models`
(SQLAlchemy models), and `core` (settings).

## Technology stack

- Python 3.10+ / FastAPI / Pydantic v2
- SQLAlchemy (async) + SQLite (aiosqlite)
- PyMuPDF (PDF extraction)
- sentence-transformers (`all-MiniLM-L6-v2`)
- Qdrant (`qdrant-client`)
- Redis (`redis` asyncio client; `fakeredis` in tests)
- OpenAI SDK against any OpenAI-compatible endpoint
- pytest + pytest-asyncio + httpx (ASGI transport)
- Ruff

## API endpoints

### GET /health

Liveness probe. Returns `200` with `{"status": "ok"}`.

### POST /api/v1/documents

Ingest a PDF or TXT document: extract text, chunk it, persist metadata in
SQLite, embed the chunks, and store the vectors in Qdrant.

Multipart form fields:

| Field | Type | Required | Default | Notes |
| ----- | ---- | -------- | ------- | ----- |
| `file` | file | yes | – | PDF or TXT, max 25 MB (`MAX_UPLOAD_SIZE`) |
| `chunk_strategy` | string | yes | – | `recursive` or `sentence` |
| `chunk_size` | int | no | 1000 | Maximum characters per chunk |
| `overlap` | int | no | 200 | Characters shared between adjacent chunks |

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -F "file=@notes.txt" \
  -F "chunk_strategy=recursive" \
  -F "chunk_size=1000" \
  -F "overlap=200"
```

Response `201`:

```json
{
  "id": 1,
  "filename": "notes.txt",
  "file_type": "txt",
  "chunk_strategy": "recursive",
  "chunk_count": 3,
  "created_at": "2026-10-07T12:00:00Z"
}
```

Errors: `400` unsupported file type or empty/invalid file, `413` file exceeds
the size limit, `422` invalid form values, `502` embedding or vector-store
failure (failed ingestions roll back and clean up partial vectors).

### POST /api/v1/chat

Conversational RAG endpoint. The `conversation_id` identifies the
conversation: history is loaded from Redis and replayed into the LLM prompt,
and the completed user/assistant turn is stored back in Redis after a
successful answer.

Request:

```json
{
  "conversation_id": "demo-123",
  "message": "What is RAG?"
}
```

Validation: `conversation_id` is required and must not be blank; `message`
must be 1–4000 characters and must not be blank.

Response `200`:

```json
{
  "conversation_id": "demo-123",
  "answer": "RAG combines retrieval with generation...",
  "sources": [
    {"filename": "guide.pdf", "chunk_index": 2, "score": 0.81}
  ]
}
```

`sources` lists the retrieved chunks cited for the answer (empty when no
relevant context was found). Errors: `422` invalid request, `502` retrieval
or LLM failure, `500` booking persistence failure or unexpected internal
error.

## Interview booking

Interview booking is handled through the conversational API using an LLM
function call; there is no separate booking endpoint:

1. The user asks to schedule an interview in a chat message.
2. The LLM invokes the `book_interview` tool with `name`, `email`,
   `date` (`YYYY-MM-DD`), and `time` (`HH:MM`).
3. Tool arguments are validated with the `InterviewBookingCreate` Pydantic
   schema (non-blank name, valid email format, strict date/time parsing).
4. `BookingService` persists the booking in the SQLite `interview_bookings`
   table.
5. The confirmation is generated from the persisted booking, e.g.
   "Your interview has been booked for 2026-10-10 at 14:00." (example
   date/time).
6. The successful user/assistant turn is stored in Redis.

Invalid or incomplete tool arguments create no database record; the assistant
asks for the missing or corrected information instead (a normal `200`
response, not an HTTP validation error). Identical bookings (same name,
email, date, and time) return the existing record instead of creating a
duplicate. If persistence fails, the request returns an error and nothing is
stored in Redis. No external calendar system is integrated.

## Environment configuration

All settings have defaults, so the application starts without any
configuration. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Purpose |
| -------- | ------- |
| `APP_NAME`, `APP_DEBUG` | Application name and debug flag |
| `MAX_UPLOAD_SIZE` | Maximum upload size in bytes (default 25 MB) |
| `DATABASE_URL` | SQLAlchemy async URL (default: local SQLite in `data/`) |
| `QDRANT_MODE` | `local` (default, embedded) or `server` (remote) |
| `QDRANT_LOCAL_PATH` | Storage path for Qdrant local mode (default `./data/qdrant`) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Remote Qdrant connection (used in `server` mode) |
| `QDRANT_COLLECTION_NAME` | Collection for document vectors (default `documents`) |
| `REDIS_URL` | Redis connection for conversation memory |
| `REDIS_TTL_SECONDS` | Conversation history TTL in seconds (default 86400) |
| `REDIS_MAX_HISTORY` | Messages kept per conversation (default 50) |
| `RAG_TOP_K` | Chunks retrieved per query (default 5) |
| `RAG_SCORE_THRESHOLD` | Minimum similarity score (default 0.0 = accept all results) |
| `RAG_MAX_CONTEXT_CHARS` | Maximum retrieved-context characters in the prompt (default 12000) |
| `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE` | Embedding model and device (default `all-MiniLM-L6-v2`, `cpu`) |
| `LLM_PROVIDER` | Provider identifier (default `openai_compatible`, the implemented provider) |
| `LLM_BASE_URL` | OpenAI-compatible endpoint (default: Ollama at `http://localhost:11434/v1`) |
| `LLM_API_KEY` | API key for hosted providers; placeholder key used for local endpoints |
| `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_TIMEOUT` | Model name, sampling temperature, request timeout |

Required to run end-to-end: a running Redis server (default
`redis://localhost:6379/0`) and a reachable OpenAI-compatible LLM endpoint
(default: local Ollama). `LLM_API_KEY` is only required for hosted providers
such as OpenAI or Groq; when left empty, a placeholder key is used, which
local endpoints like Ollama accept. Qdrant runs in embedded local mode by
default, so no Qdrant server is needed for local development. Do not commit
real API keys; `.env` is git-ignored and `.env.example` holds placeholders
only.

## Local setup

Windows PowerShell (use `cp` instead of `Copy-Item` in Git Bash):

```powershell
git clone <repository-url>
cd ai-rag-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env if needed
```

Optional: install the CPU-only PyTorch build first to avoid the large CUDA
download: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The embedding model is downloaded once from Hugging Face on the first
ingestion or chat request; later runs use the local cache.

## Running the API

```powershell
uvicorn app.main:app --reload
```

The API is served at `http://localhost:8000`. FastAPI's automatically
generated interactive documentation is available at `/docs` (and `/redoc`);
these are generated API references, not a custom frontend.

## Testing

```powershell
python -m pytest -q
ruff check .
```

The suite currently contains 188 tests covering extraction, chunking,
embeddings, the Qdrant vector store, Redis memory, the LLM provider, the
RAG pipeline, booking, and the API endpoints. Tests use fakes/mocks,
`fakeredis`, and temporary local Qdrant/SQLite directories; they do not call
a real LLM and do not require a running Redis or Qdrant server. One
end-to-end test loads the real local embedding model. No production secrets
are needed or exposed.

## Project structure

```
app/
├── api/               # FastAPI routes and dependency wiring
│   └── v1/            # /api/v1 documents and chat endpoints
├── core/              # Settings (environment configuration)
├── models/            # SQLAlchemy models (documents, chunks, bookings)
├── ports/             # Provider-agnostic interfaces
├── schemas/           # Pydantic request/response models
└── services/          # Business logic
    ├── booking/       # Booking service and book_interview tool
    ├── documents/     # Extraction, chunking, ingestion
    ├── embeddings/    # Sentence-transformer embeddings
    ├── llm/           # OpenAI-compatible provider
    ├── memory/        # Redis conversation memory
    ├── rag/           # RAG orchestration
    └── vector_store/  # Qdrant vector store
tests/                 # pytest suite
```

Runtime data (local SQLite database and Qdrant storage) is written to `data/`
and is git-ignored.

## Implementation constraints

The implementation intentionally does not use:

- FAISS
- Chroma / ChromaDB
- `RetrievalQAChain`
- LangChain

No custom frontend/UI is included; the project scope covers backend REST
APIs only, and FastAPI's generated `/docs` is the only interactive surface.

## Design decisions

- **Qdrant** selected as the vector database (embedded local mode for
  development, remote server mode supported).
- **SQLite** via async SQLAlchemy for document metadata and bookings —
  simple, zero-configuration persistence.
- **sentence-transformers** (`all-MiniLM-L6-v2`) for local, normalized
  embeddings.
- **OpenAI-compatible interface** keeps the LLM provider replaceable
  (Ollama, OpenAI, Groq, and similar endpoints).
- **Redis** for conversation history, with per-conversation history limits
  and TTL.
- **Ports/interfaces** separate orchestration from infrastructure
  implementations.
- **Explicit RAG orchestration** in `RAGService` rather than delegation to
  a prebuilt chain.

## Current status

The backend is fully implemented and tested. Both required REST APIs —
document ingestion and conversational RAG — are working, including multi-turn
memory and LLM-driven interview booking with validation, persistence, and
duplicate protection. The test suite (188 tests) and Ruff linting pass
cleanly.
