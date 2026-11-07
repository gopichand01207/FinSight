"""
FastAPI Router — Query / Q&A Endpoints
  POST /api/v1/query  → answer a question about indexed financial documents
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logging import get_logger
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag import RAGService, get_rag_service

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question about indexed financial documents",
    description="""
Runs the full FinSight RAG pipeline:

1. **Hybrid Retrieval** — Dense (OpenAI embeddings via ChromaDB) + Sparse (BM25) results
   fused with Reciprocal Rank Fusion (RRF).
2. **Grounded Generation** — GPT-4o answers strictly from retrieved context chunks.
3. **Citations** — Every response includes ranked source citations with page numbers,
   section labels, and relevance scores.

Set `temperature=0` (default) for factual, deterministic financial analysis.
Optionally restrict search to specific `document_ids`.
""",
)
async def query_documents(
    request: QueryRequest,
    rag_svc: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Main Q&A endpoint. Accepts a natural-language question and returns a
    grounded answer with source citations from the indexed financial documents.
    """
    try:
        response = rag_svc.answer(request)
        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("RAG pipeline error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(exc)}",
        ) from exc

