"""
FinSight FastAPI Application Entry Point
─────────────────────────────────────────
Initialises the FastAPI app with:
  - CORS middleware
  - API routers (documents, query, health)
  - Startup / shutdown lifecycle events
  - Custom OpenAPI metadata
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, health, query
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

configure_logging(debug=settings.debug)
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    - Startup:  pre-warms vector store connection + BM25 index
    - Shutdown: flushes ChromaDB write buffer
    """
    logger.info("FinSight API starting up", version=settings.app_version)

    # Pre-warm connections
    from app.services.ingestion import get_vector_store
    from app.services.retrieval import get_bm25_retriever, get_vector_store as _vs

    try:
        vs = get_vector_store()
        count = vs._collection.count()
        logger.info("ChromaDB connected", indexed_chunks=count)

        # Build in-memory BM25 index from existing ChromaDB corpus
        if count > 0:
            bm25 = get_bm25_retriever()
            results = vs._collection.get(include=["documents", "metadatas"])
            from langchain_core.documents import Document
            docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                )
            ]
            bm25.build_index(docs)
            logger.info("BM25 index ready", corpus_size=len(docs))
    except Exception as exc:
        logger.warning("Pre-warm failed (continuing anyway)", error=str(exc))

    yield  # ← app runs here

    logger.info("FinSight API shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinSight API",
    description=(
        "AI-powered financial document intelligence platform. "
        "Upload SEC filings, annual reports, or earnings releases and ask "
        "natural-language questions with grounded, cited answers."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }

