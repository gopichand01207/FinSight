# FinSight: AI-Powered Financial Document Intelligence

> **Stack**: LangChain · OpenAI GPT-4o · text-embedding-3-large · ChromaDB · FastAPI · Streamlit · Docker · AWS EC2 · GitHub Actions

---

## Overview

**FinSight** is a production-grade Retrieval-Augmented Generation (RAG) platform that transforms complex financial documents — SEC filings (10-K, 10-Q, 8-K), annual reports, and earnings releases — into an interactive, cited Q&A experience.

The system reduces analyst document review time by **~70%** by enabling natural-language queries with grounded, source-cited answers over large financial document corpora.

---

## Architecture

```
 PDF Upload (FastAPI)
       │
       ▼
 PyMuPDF Parser ──────► Page text + section metadata
       │
       ▼
 SemanticChunker ──────► Sentence-aware chunks (100–512 tokens)
       │                  with 50-token overlap
       ▼
 OpenAI Embeddings ────► text-embedding-3-large (3,072-dim dense vectors)
       │
       ▼
 ChromaDB ─────────────► Persistent cosine-similarity vector index
       │
       ▼
 Hybrid Retrieval:
   ├─ Dense  ─── ChromaDB cosine similarity (top-10)
   └─ Sparse ─── BM25/Okapi (rank_bm25, top-10)
         │
         ▼
 Reciprocal Rank Fusion (RRF k=60) → top-5 chunks
         │
         ▼
 GPT-4o Generation ────► Grounded answer + [SOURCE-N] citations
         │
         ▼
 QueryResponse (JSON) → Streamlit Dashboard / API client
```

---

## Key Technical Features

### 1. Semantic Chunking
- Sentence-boundary detection via regex
- Greedy token-budget enforcement (min: 100 tokens, max: 512 tokens)
- 50-token overlap window to preserve cross-chunk context
- MD5-keyed chunk IDs for idempotent re-ingestion

### 2. Hybrid Dense-Sparse Retrieval
- **Dense**: OpenAI `text-embedding-3-large` (3,072 dimensions) + ChromaDB cosine similarity
- **Sparse**: BM25/Okapi (`rank_bm25`) keyword frequency scoring
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60) merges both ranked lists
- Final scores normalised to `[0, 1]` for consistent citation relevance display

### 3. Grounded RAG Generation
- LangChain LCEL pipeline: `RunnablePassthrough | ChatPromptTemplate | ChatOpenAI | StrOutputParser`
- System prompt enforces citation-only answers (`[SOURCE-N]` tags)
- Temperature=0 for deterministic financial analysis
- Full latency breakdown: retrieval + generation + total

### 4. FastAPI Backend
- Async endpoints with Pydantic v2 validation
- `/api/v1/documents/upload` — PDF ingestion with file-size guard
- `/api/v1/query` — hybrid RAG Q&A with structured JSON response
- `/api/v1/health` — ChromaDB + OpenAI API health checks
- CORS middleware configured for Streamlit origin

### 5. Streamlit Dashboard
- 4-panel UI: Document Upload · Q&A Chat · Analytics · Settings
- Drag-and-drop PDF ingestion with real-time progress
- Citation explorer with page numbers, section labels, relevance scores
- Query latency time-series chart
- Live API health badge in sidebar

---

## Project Structure

```
FinSight/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── core/
│   │   ├── config.py            # Pydantic settings (env-driven)
│   │   └── logging.py           # Structlog JSON/console logging
│   ├── api/
│   │   ├── documents.py         # Upload + list endpoints
│   │   ├── query.py             # RAG Q&A endpoint
│   │   └── health.py            # Health check endpoint
│   ├── services/
│   │   ├── ingestion.py         # PDF parse + semantic chunk + embed
│   │   ├── retrieval.py         # Hybrid dense-sparse-RRF retrieval
│   │   └── rag.py               # End-to-end RAG pipeline
│   └── models/
│       └── schemas.py           # Pydantic v2 request/response schemas
├── streamlit_app/
│   └── main.py                  # Multi-page Streamlit dashboard
├── tests/
│   └── test_finsight.py         # Unit + integration tests (pytest)
├── docker/
│   └── Dockerfile               # Multi-stage production Docker build
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions CI/CD pipeline
├── scripts/
│   ├── ingest_sample.py         # CLI PDF ingestion tool
│   └── query_cli.py             # Interactive CLI Q&A tool
├── docker-compose.yml           # Local development stack
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Dev/test dependencies
├── pyproject.toml               # Ruff + mypy + pytest config
└── .env.example                 # Environment variable template
```

---

## Setup & Run

### Local Development

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 2. Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Start the FastAPI backend
uvicorn app.main:app --reload --port 8000

# 4. Start the Streamlit dashboard (in a new terminal)
streamlit run streamlit_app/main.py
```

### Docker (recommended)

```bash
# Start the full stack (API + Streamlit)
docker compose up --build

# Access:
#   API docs:  http://localhost:8000/docs
#   Dashboard: http://localhost:8501
```

### Run Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | System health check |
| `POST` | `/api/v1/documents/upload` | Upload + index a PDF |
| `GET` | `/api/v1/documents/` | List all indexed documents |
| `POST` | `/api/v1/query` | RAG Q&A with citations |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apple'\''s total revenue in fiscal year 2024?",
    "top_k": 5,
    "temperature": 0.0
  }'
```

**Response:**
```json
{
  "question": "What was Apple's total revenue in fiscal year 2024?",
  "answer": "Apple's total revenue for fiscal year 2024 was $391.0 billion [SOURCE-1], ...",
  "citations": [
    {
      "chunk_id": "a3f8c...",
      "filename": "apple_10k_2024.pdf",
      "page_number": 24,
      "section": "Income Statement",
      "relevance_score": 0.9823,
      "text_snippet": "Net sales for fiscal 2024 were $391.0 billion..."
    }
  ],
  "retrieval_latency_ms": 38.4,
  "generation_latency_ms": 1240.6,
  "total_latency_ms": 1279.0,
  "model_used": "gpt-4o",
  "tokens_used": 418
}
```

---

## CI/CD Pipeline (GitHub Actions)

Three-stage pipeline on every push to `main`:

1. **Lint & Test** — Ruff linting + mypy type checks + pytest with coverage
2. **Build & Push** — Multi-stage Docker image pushed to GitHub Container Registry
3. **Deploy** — SSH into AWS EC2, pull latest image, restart via Docker Compose, health-check

---

## Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| LLM | OpenAI GPT-4o | Answer generation |
| Embeddings | text-embedding-3-large | Dense vector encoding (3,072-dim) |
| RAG Framework | LangChain LCEL | Pipeline orchestration |
| Vector DB | ChromaDB | Persistent dense index |
| Sparse Retrieval | rank_bm25 | BM25/Okapi keyword search |
| PDF Parsing | PyMuPDF (fitz) | High-fidelity text extraction |
| Tokenizer | tiktoken | Token counting (cl100k_base) |
| Backend | FastAPI + Uvicorn | Async REST API |
| Frontend | Streamlit | Interactive dashboard |
| Containerization | Docker (multi-stage) | Reproducible builds |
| Orchestration | Docker Compose | Local + production deployment |
| Cloud | AWS EC2 | Production hosting |
| CI/CD | GitHub Actions | Automated test/build/deploy |
| Logging | structlog | Structured JSON logging |

---

*Built by Bhukya Gopi Chand Naik — Resume Project Portfolio 2026*

