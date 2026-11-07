"""
FinSight RAG (Retrieval-Augmented Generation) Service
───────────────────────────────────────────────────────
Orchestrates the full RAG pipeline:
  1. Hybrid retrieval (dense + sparse → RRF)
  2. Prompt construction with retrieved context + citations
  3. OpenAI GPT-4o generation
  4. Structured QueryResponse assembly

The system prompt enforces grounding — the LLM is explicitly
instructed to answer only from the retrieved context and to cite
each supporting chunk by its [SOURCE-N] tag.
"""
from __future__ import annotations

import time
from typing import Optional
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import Citation, QueryRequest, QueryResponse
from app.services.retrieval import get_retrieval_service

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI assistant.
Your task is to answer questions about financial documents (SEC filings, annual reports, earnings releases) with precision and clarity.

INSTRUCTIONS:
- Answer ONLY using the provided context chunks below. Do NOT use external knowledge.
- For every factual claim in your answer, cite the source chunk using [SOURCE-N] notation.
- If the context does not contain enough information to answer confidently, say so explicitly.
- Structure your answer clearly: use bullet points or numbered lists for multi-part answers.
- For numerical data (revenue, EPS, margins), always include the exact figure and time period.
- Keep your answer concise but complete — typically 2–5 paragraphs.

CONTEXT CHUNKS:
{context}
"""

HUMAN_TEMPLATE = "Question: {question}"

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
)


# ─────────────────────────────────────────────────────────────────────────────
# RAG SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class RAGService:
    """
    End-to-end RAG pipeline service.
    Uses LangChain LCEL (LangChain Expression Language) runnables.
    """

    def __init__(self) -> None:
        self.retrieval_service = get_retrieval_service()
        self._llm_cache: dict[float, ChatOpenAI] = {}

    def _get_llm(self, temperature: float) -> ChatOpenAI:
        """Lazily create ChatOpenAI instances keyed by temperature."""
        if temperature not in self._llm_cache:
            self._llm_cache[temperature] = ChatOpenAI(
                model=settings.openai_model,
                temperature=temperature,
                openai_api_key=settings.openai_api_key,
                max_tokens=1500,
            )
        return self._llm_cache[temperature]

    def _build_context_string(
        self, retrieved: list[tuple]
    ) -> tuple[str, list[Citation]]:
        """
        Convert (Document, score) pairs into a numbered context string
        and a parallel list of Citation objects.
        """
        context_parts: list[str] = []
        citations: list[Citation] = []

        for idx, (doc, score) in enumerate(retrieved, start=1):
            meta = doc.metadata
            source_tag = f"[SOURCE-{idx}]"
            snippet = doc.page_content[:400]

            context_parts.append(
                f"{source_tag}\n"
                f"File: {meta.get('filename', 'unknown')} | "
                f"Page: {meta.get('page_number', '?')} | "
                f"Section: {meta.get('section', 'General')}\n"
                f"{snippet}\n"
            )

            citations.append(
                Citation(
                    chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", ""),
                    page_number=meta.get("page_number"),
                    section=meta.get("section"),
                    relevance_score=round(float(score), 4),
                    text_snippet=snippet,
                )
            )

        return "\n---\n".join(context_parts), citations

    def answer(self, request: QueryRequest) -> QueryResponse:
        """
        Run the full RAG pipeline for a user query.
        """
        logger.info("RAG query received", question=request.question[:80])

        # ── 1. Hybrid retrieval ──────────────────────────────────────────────
        t_ret_start = time.perf_counter()
        document_ids = (
            [str(did) for did in request.document_ids]
            if request.document_ids
            else None
        )
        retrieved = self.retrieval_service.retrieve(
            query=request.question,
            top_k=request.top_k or settings.hybrid_top_k,
            document_ids=document_ids,
        )
        retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000

        # ── 2. Build context + citations ─────────────────────────────────────
        context_str, citations = self._build_context_string(retrieved)

        # ── 3. LLM generation ────────────────────────────────────────────────
        t_gen_start = time.perf_counter()
        llm = self._get_llm(request.temperature)

        chain = (
            RunnablePassthrough()
            | _PROMPT
            | llm
            | StrOutputParser()
        )

        answer_text = chain.invoke(
            {"context": context_str, "question": request.question}
        )
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000
        total_latency_ms = retrieval_latency_ms + generation_latency_ms

        # Approximate token count (4 chars ≈ 1 token for English text)
        tokens_estimate = (len(context_str) + len(answer_text)) // 4

        logger.info(
            "RAG answer generated",
            retrieval_ms=round(retrieval_latency_ms, 2),
            generation_ms=round(generation_latency_ms, 2),
            total_ms=round(total_latency_ms, 2),
            citations=len(citations),
        )

        return QueryResponse(
            question=request.question,
            answer=answer_text,
            citations=citations,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            model_used=settings.openai_model,
            tokens_used=tokens_estimate,
        )


# Singleton
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

