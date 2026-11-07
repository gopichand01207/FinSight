"""
FinSight Hybrid Retrieval Service
───────────────────────────────────
Implements a hybrid dense-sparse retrieval pipeline:

  Dense  : OpenAI text-embedding-3-large  →  ChromaDB cosine similarity
  Sparse : BM25 (rank_bm25)               →  keyword/token frequency scoring
  Fusion : Reciprocal Rank Fusion (RRF)   →  unified ranked list

RRF formula:
  score(d) = Σ  1 / (k + rank_i(d))
  where k=60 is the standard constant that dampens high-rank dominance.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ingestion import get_vector_store

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DENSE RETRIEVAL  (ChromaDB embedding search)
# ─────────────────────────────────────────────────────────────────────────────

def dense_retrieve(
    query: str,
    top_k: int = settings.dense_top_k,
    document_ids: Optional[list[str]] = None,
) -> list[Document]:
    """
    Semantic (dense) retrieval using ChromaDB with OpenAI embeddings.
    Optionally filters to a subset of document_ids.
    """
    vector_store = get_vector_store()

    where_filter = None
    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter=where_filter,
    )

    # Attach retrieval scores to metadata
    docs = []
    for doc, score in results:
        doc.metadata["dense_score"] = round(float(score), 6)
        docs.append(doc)

    logger.debug("Dense retrieval", query_preview=query[:60], n_results=len(docs))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE RETRIEVAL  (BM25 keyword search)
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    In-memory BM25 retriever built on top of the ChromaDB corpus.
    The corpus is loaded once per session; rebuild if new documents are added.
    """

    def __init__(self) -> None:
        self._corpus: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase whitespace tokenizer — fast and effective for BM25."""
        return text.lower().split()

    def build_index(self, documents: list[Document]) -> None:
        """Build the BM25 index from a list of LangChain Documents."""
        self._corpus = documents
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built", corpus_size=len(documents))

    def retrieve(self, query: str, top_k: int = settings.sparse_top_k) -> list[Document]:
        if self._bm25 is None or not self._corpus:
            logger.warning("BM25 index not built; returning empty results")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair (score, Document) and sort descending
        ranked = sorted(
            zip(scores, self._corpus), key=lambda x: x[0], reverse=True
        )[:top_k]

        results = []
        for score, doc in ranked:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "sparse_score": round(float(score), 6)},
            )
            results.append(doc_copy)

        logger.debug("BM25 retrieval", n_results=len(results))
        return results


# Module-level singleton BM25 retriever
_bm25_retriever: BM25Retriever = BM25Retriever()


def get_bm25_retriever() -> BM25Retriever:
    return _bm25_retriever


# ─────────────────────────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = settings.rrf_k,
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked document lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each sub-list is a ranked result set (index 0 = best).
        k: RRF damping constant (default 60).

    Returns:
        List of (Document, rrf_score) sorted descending by RRF score.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:64])
            rrf_scores[chunk_id] += 1.0 / (k + rank)
            doc_map[chunk_id] = doc

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[cid], score) for cid, score in merged]


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetrievalService:
    """
    Combines dense (embedding) + sparse (BM25) retrieval via RRF.
    Normalises the final RRF scores to [0, 1] for downstream use.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = settings.hybrid_top_k,
        document_ids: Optional[list[str]] = None,
    ) -> list[tuple[Document, float]]:
        """
        Run hybrid retrieval and return top_k (Document, relevance_score) tuples.
        """
        t0 = time.perf_counter()

        # 1. Dense retrieval
        dense_results = dense_retrieve(
            query, top_k=settings.dense_top_k, document_ids=document_ids
        )

        # 2. Sparse retrieval
        bm25 = get_bm25_retriever()
        sparse_results = bm25.retrieve(query, top_k=settings.sparse_top_k)

        # 3. Fuse with RRF
        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=settings.rrf_k,
        )[:top_k]

        # 4. Normalise scores to [0, 1]
        if fused:
            max_score = fused[0][1] if fused[0][1] > 0 else 1.0
            fused = [(doc, score / max_score) for doc, score in fused]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Hybrid retrieval complete",
            query_preview=query[:60],
            n_dense=len(dense_results),
            n_sparse=len(sparse_results),
            n_fused=len(fused),
            latency_ms=round(elapsed_ms, 2),
        )

        return fused


# Singleton
_retrieval_service: HybridRetrievalService | None = None


def get_retrieval_service() -> HybridRetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = HybridRetrievalService()
    return _retrieval_service

