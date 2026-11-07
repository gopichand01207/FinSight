"""
FastAPI Router — Document Ingestion Endpoints
  POST /api/v1/documents/upload  → ingest a PDF financial document
  GET  /api/v1/documents/        → list all indexed documents
  DELETE /api/v1/documents/{id}  → remove a document from the index
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import DocumentListResponse, DocumentUploadResponse
from app.services.ingestion import get_ingestion_service, DocumentIngestionService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a financial document (PDF)",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file of the financial report or SEC filing"),
    settings: Settings = Depends(get_settings),
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    """
    Upload endpoint:
      1. Validates file type (PDF only).
      2. Saves the file to the upload directory.
      3. Triggers the ingestion pipeline (parse → chunk → embed → store).
      4. Returns document metadata and chunk statistics.
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted. Please upload a valid financial document.",
        )

    file_size_bytes = 0
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    dest_path = upload_dir / f"{document_id}_{file.filename}"

    # ── Save file ────────────────────────────────────────────────────────────
    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                file_size_bytes += len(chunk)
                if file_size_bytes > settings.max_upload_size_mb * 1024 * 1024:
                    buffer.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.max_upload_size_mb} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File save failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    # ── Ingest ───────────────────────────────────────────────────────────────
    try:
        result = ingestion_svc.ingest(file_path=dest_path, document_id=document_id)
    except Exception as exc:
        logger.error("Ingestion failed", document_id=document_id, error=str(exc))
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}",
        ) from exc

    logger.info(
        "Document uploaded and indexed",
        document_id=document_id,
        filename=file.filename,
        num_chunks=result["num_chunks"],
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        num_chunks=result["num_chunks"],
        message=f"Successfully parsed and indexed {result['num_chunks']} semantic chunks in {result['latency_ms']:.0f} ms.",
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all indexed financial documents",
)
async def list_documents(
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    """
    Returns metadata for all documents currently indexed in ChromaDB.
    """
    try:
        collection = ingestion_svc.vector_store._collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) or []

        # Deduplicate by document_id
        seen: dict[str, dict] = {}
        for meta in metadatas:
            did = meta.get("document_id", "unknown")
            if did not in seen:
                seen[did] = {
                    "document_id": did,
                    "filename": meta.get("filename", ""),
                    "num_chunks": 0,
                    "ingested_at": None,
                }
            seen[did]["num_chunks"] += 1

        from datetime import datetime
        items = [
            {
                "document_id": v["document_id"],
                "filename": v["filename"],
                "num_chunks": v["num_chunks"],
                "ingested_at": datetime.utcnow(),
            }
            for v in seen.values()
        ]

        return DocumentListResponse(documents=items, total=len(items))
    except Exception as exc:
        logger.error("Failed to list documents", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list.",
        ) from exc

