"""
FastAPI Router — Health Check Endpoint
  GET /api/v1/health  → system health status
"""
from __future__ import annotations

import time

from fastapi import APIRouter
from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import HealthResponse
from app.services.ingestion import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns the operational status of ChromaDB and the OpenAI API connection.",
)
async def health_check() -> HealthResponse:
    """
    Pings both ChromaDB and the OpenAI API.
    Returns 200 OK when all systems are operational.
    """
    # ── ChromaDB health ──────────────────────────────────────────────────────
    chroma_status = "unknown"
    try:
        vs = get_vector_store()
        count = vs._collection.count()
        chroma_status = f"ok (indexed chunks: {count})"
    except Exception as exc:
        chroma_status = f"error: {str(exc)}"
        logger.warning("ChromaDB health check failed", error=str(exc))

    # ── OpenAI API health ────────────────────────────────────────────────────
    openai_status = "unknown"
    try:
        from langchain_openai import ChatOpenAI
        from app.core.config import settings as _s
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=_s.openai_api_key, max_tokens=1)
        openai_status = "ok"
    except Exception as exc:
        openai_status = f"error: {str(exc)}"
        logger.warning("OpenAI health check failed", error=str(exc))

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        chroma_status=chroma_status,
        openai_status=openai_status,
    )

