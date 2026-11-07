"""
Pydantic v2 request / response schemas for FinSight API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─────────────────────────── Document schemas ───────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Returned after a successful document ingest."""
    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    num_chunks: int
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "indexed"
    message: str = "Document successfully parsed, chunked, and indexed."


class DocumentListItem(BaseModel):
    document_id: UUID
    filename: str
    num_chunks: int
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


# ─────────────────────────── Citation schema ────────────────────────────────

class Citation(BaseModel):
    """
    Source citation attached to each answer.
    Maps directly to a ChromaDB chunk and its originating document.
    """
    chunk_id: str = Field(..., description="Unique ChromaDB chunk identifier")
    document_id: str
    filename: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Hybrid RRF-normalised relevance score"
    )
    text_snippet: str = Field(
        ..., max_length=400,
        description="Verbatim excerpt from the source chunk"
    )


# ─────────────────────────── QA schemas ─────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload for POST /api/v1/query"""
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Natural-language question about ingested financial documents",
        examples=["What was Apple's total revenue in Q3 2024?"],
    )
    document_ids: Optional[list[UUID]] = Field(
        default=None,
        description="Restrict retrieval to these document IDs. None = search all.",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve (overrides server default)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM sampling temperature (0 = deterministic, grounded answers)",
    )


class QueryResponse(BaseModel):
    """Full structured response from the RAG pipeline."""
    question: str
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved chunks")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ordered list of source citations supporting the answer",
    )
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model_used: str
    tokens_used: int


# ─────────────────────────── Health schema ───────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    chroma_status: str
    openai_status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

