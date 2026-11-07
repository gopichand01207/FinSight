"""
FinSight Document Ingestion Service
────────────────────────────────────
Handles:
  1. PDF / text parsing with PyMuPDF
  2. Semantic chunking (sentence-aware with token-budget enforcement)
  3. OpenAI text-embedding-3-large embedding generation
  4. ChromaDB vector persistence
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Generator

import pymupdf as fitz  # PyMuPDF (new import style)
import tiktoken
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Tokenizer (cl100k_base used by text-embedding-3-large) ─────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────────────────────────
# PDF PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from a PDF using PyMuPDF, preserving page metadata.
    Returns a list of page dicts: {page_number, text, section}.
    """
    logger.info("Parsing PDF", path=str(file_path))
    doc = fitz.open(str(file_path))
    pages: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        # Strip excessive whitespace while preserving paragraph structure
        cleaned = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
        if cleaned:
            pages.append(
                {
                    "page_number": page_num,
                    "text": cleaned,
                    "section": _infer_section(cleaned),
                }
            )

    doc.close()
    logger.info("PDF parsed", pages=len(pages))
    return pages


def _infer_section(text: str) -> str:
    """
    Heuristic: infer financial document section from leading lines.
    Maps common SEC filing / annual report headings.
    """
    heading_map = {
        "management": "Management Discussion & Analysis",
        "risk factor": "Risk Factors",
        "financial statement": "Financial Statements",
        "balance sheet": "Balance Sheet",
        "income statement": "Income Statement",
        "cash flow": "Cash Flow Statement",
        "note": "Notes to Financial Statements",
        "market risk": "Quantitative Market Risk",
        "executive": "Executive Compensation",
    }
    first_line = text.split("\n")[0].lower()
    for keyword, section_name in heading_map.items():
        if keyword in first_line:
            return section_name
    return "General"


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Sentence-aware chunker that respects min/max token budgets.
    Sentence boundaries are detected via regex; chunks are grown
    greedily until the max-token ceiling is reached, then a new
    chunk is started with an overlap window.
    """

    _SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9\$\"])",
    )

    def __init__(
        self,
        min_tokens: int = settings.semantic_chunk_min_tokens,
        max_tokens: int = settings.semantic_chunk_max_tokens,
        overlap_tokens: int = settings.chunk_overlap_tokens,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _split_sentences(self, text: str) -> list[str]:
        raw = self._SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences

    def chunk(
        self,
        text: str,
        page_number: int = 0,
        section: str = "General",
        document_id: str = "",
        filename: str = "",
    ) -> list[Document]:
        sentences = self._split_sentences(text)
        chunks: list[Document] = []
        current_sentences: list[str] = []
        current_tokens = 0
        overlap_buffer: list[str] = []

        def _flush(sents: list[str]) -> None:
            chunk_text = " ".join(sents).strip()
            if _count_tokens(chunk_text) >= self.min_tokens:
                chunk_id = hashlib.md5(
                    f"{document_id}:{page_number}:{chunk_text[:64]}".encode()
                ).hexdigest()
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "filename": filename,
                            "page_number": page_number,
                            "section": section,
                            "token_count": _count_tokens(chunk_text),
                        },
                    )
                )

        for sentence in sentences:
            sent_tokens = _count_tokens(sentence)
            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                _flush(current_sentences)
                # Build overlap: take sentences from the end of current chunk
                overlap_buffer = []
                overlap_tok = 0
                for s in reversed(current_sentences):
                    t = _count_tokens(s)
                    if overlap_tok + t <= self.overlap_tokens:
                        overlap_buffer.insert(0, s)
                        overlap_tok += t
                    else:
                        break
                current_sentences = overlap_buffer.copy()
                current_tokens = overlap_tok

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            _flush(current_sentences)

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING + VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns an OpenAIEmbeddings instance configured for
    text-embedding-3-large (3072-dim dense vectors).
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        openai_api_key=settings.openai_api_key,
        chunk_size=256,  # batch size for embedding API calls
    )


def get_vector_store() -> Chroma:
    """Return (or create) the persistent ChromaDB vector store."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class DocumentIngestionService:
    """
    Orchestrates the full ingest pipeline:
      parse → semantic chunk → embed → persist to ChromaDB
    """

    def __init__(self) -> None:
        self.chunker = SemanticChunker()
        self.vector_store = get_vector_store()

    def ingest(self, file_path: Path, document_id: str | None = None) -> dict:
        """
        Ingest a PDF financial document.

        Returns:
            dict with document_id, filename, num_chunks, latency_ms
        """
        t0 = time.perf_counter()
        document_id = document_id or str(uuid.uuid4())
        filename = file_path.name

        # 1. Parse
        pages = parse_pdf(file_path)

        # 2. Chunk
        all_chunks: list[Document] = []
        for page in pages:
            chunks = self.chunker.chunk(
                text=page["text"],
                page_number=page["page_number"],
                section=page["section"],
                document_id=document_id,
                filename=filename,
            )
            all_chunks.extend(chunks)

        logger.info("Chunking complete", num_chunks=len(all_chunks), document_id=document_id)

        # 3. Embed + persist to ChromaDB (batched)
        texts = [c.page_content for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        ids = [c.metadata["chunk_id"] for c in all_chunks]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Ingestion complete",
            document_id=document_id,
            num_chunks=len(all_chunks),
            latency_ms=round(elapsed_ms, 2),
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "num_chunks": len(all_chunks),
            "latency_ms": round(elapsed_ms, 2),
        }


# Singleton — shared across request handlers
_ingestion_service: DocumentIngestionService | None = None


def get_ingestion_service() -> DocumentIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DocumentIngestionService()
    return _ingestion_service

