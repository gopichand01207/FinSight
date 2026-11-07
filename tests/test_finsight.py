"""
Unit and Integration Tests for FinSight
Tests:
  - Semantic chunker correctness
  - Hybrid retrieval (mocked ChromaDB + BM25)
  - API endpoints (FastAPI TestClient)
  - RAG pipeline (mocked OpenAI)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.services.ingestion import SemanticChunker, _count_tokens, _infer_section


# ─────────────────────────────────────────────────────────────────────────────
# SemanticChunker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticChunker:
    """Tests for the sentence-aware semantic chunker."""

    def setup_method(self):
        self.chunker = SemanticChunker(
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=20,
        )

    def test_basic_chunking_produces_documents(self):
        text = (
            "Apple Inc. reported record revenue of $94.9 billion in Q4 2024. "
            "This represents a 6% increase year-over-year. "
            "iPhone sales accounted for 52% of total revenue. "
            "Services revenue grew by 12% to $24.2 billion. "
            "The company returned $29 billion to shareholders through buybacks and dividends."
        )
        chunks = self.chunker.chunk(
            text=text,
            page_number=1,
            section="Financial Statements",
            document_id="test-doc-id",
            filename="apple_10k.pdf",
        )
        assert len(chunks) >= 1, "Chunker should produce at least one chunk"

    def test_chunk_metadata_fields(self):
        text = (
            "Total revenue for fiscal year 2024 was $383.3 billion. "
            "Operating income was $114.3 billion with a margin of 29.8%."
        )
        chunks = self.chunker.chunk(
            text=text,
            page_number=5,
            section="Income Statement",
            document_id="doc-abc",
            filename="annual_report.pdf",
        )
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["document_id"] == "doc-abc"
            assert chunk.metadata["filename"] == "annual_report.pdf"
            assert chunk.metadata["page_number"] == 5
            assert chunk.metadata["section"] == "Income Statement"
            assert "token_count" in chunk.metadata

    def test_chunk_respects_max_tokens(self):
        # Create text that is definitely longer than max_tokens
        long_text = " ".join(["This is a sentence about financial performance."] * 50)
        chunks = self.chunker.chunk(
            text=long_text,
            page_number=1,
            section="General",
            document_id="doc-long",
            filename="long_doc.pdf",
        )
        for chunk in chunks:
            assert chunk.metadata["token_count"] <= self.chunker.max_tokens + 50, (
                "Chunk token count should not significantly exceed max_tokens"
            )

    def test_empty_text_returns_no_chunks(self):
        chunks = self.chunker.chunk(
            text="",
            page_number=1,
            section="General",
            document_id="doc-empty",
            filename="empty.pdf",
        )
        assert chunks == [], "Empty text should produce zero chunks"

    def test_chunk_ids_are_unique(self):
        text = " ".join([f"Sentence number {i} about financials." for i in range(30)])
        chunks = self.chunker.chunk(
            text=text, page_number=1, section="General",
            document_id="doc-uniq", filename="test.pdf",
        )
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# Section Inference Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionInference:

    def test_income_statement_detection(self):
        text = "Income Statement\nRevenue: $100M"
        assert _infer_section(text) == "Income Statement"

    def test_risk_factors_detection(self):
        text = "Risk Factors\nThe company faces competition from..."
        assert _infer_section(text) == "Risk Factors"

    def test_cash_flow_detection(self):
        text = "Cash flow from operating activities was $50 billion."
        assert _infer_section(text) == "Cash Flow Statement"

    def test_general_fallback(self):
        text = "Some random financial text without a clear heading."
        assert _infer_section(text) == "General"


# ─────────────────────────────────────────────────────────────────────────────
# Token Counter Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenCounter:

    def test_empty_string(self):
        assert _count_tokens("") == 0

    def test_known_token_count(self):
        # "Hello world" = 2 tokens in cl100k_base
        count = _count_tokens("Hello world")
        assert count == 2

    def test_long_text_has_more_tokens(self):
        short = "Revenue grew."
        long = "Revenue grew significantly in Q4 2024 driven by iPhone and Services segments."
        assert _count_tokens(long) > _count_tokens(short)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Retriever Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Retriever:

    def setup_method(self):
        from app.services.retrieval import BM25Retriever
        from langchain.schema import Document
        self.retriever = BM25Retriever()
        self.docs = [
            Document(
                page_content="Apple revenue increased by 6% to reach $94.9 billion in Q4 2024.",
                metadata={"chunk_id": "c1", "document_id": "d1", "filename": "apple.pdf",
                          "page_number": 1, "section": "Financial Statements"},
            ),
            Document(
                page_content="Risk factors include supply chain disruptions and foreign exchange volatility.",
                metadata={"chunk_id": "c2", "document_id": "d1", "filename": "apple.pdf",
                          "page_number": 10, "section": "Risk Factors"},
            ),
            Document(
                page_content="Services segment revenue was $24.2 billion, growing 12% year-over-year.",
                metadata={"chunk_id": "c3", "document_id": "d1", "filename": "apple.pdf",
                          "page_number": 5, "section": "Income Statement"},
            ),
        ]
        self.retriever.build_index(self.docs)

    def test_index_built_successfully(self):
        assert self.retriever._bm25 is not None
        assert len(self.retriever._corpus) == 3

    def test_retrieves_relevant_documents(self):
        results = self.retriever.retrieve("Apple revenue Q4 2024", top_k=2)
        assert len(results) > 0
        # The first result should contain "revenue"
        assert "revenue" in results[0].page_content.lower()

    def test_sparse_score_in_metadata(self):
        results = self.retriever.retrieve("risk supply chain", top_k=1)
        assert "sparse_score" in results[0].metadata

    def test_top_k_respected(self):
        results = self.retriever.retrieve("revenue", top_k=2)
        assert len(results) <= 2

    def test_empty_index_returns_empty(self):
        from app.services.retrieval import BM25Retriever
        empty_retriever = BM25Retriever()
        results = empty_retriever.retrieve("any query", top_k=5)
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# RRF Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReciprocalRankFusion:

    def setup_method(self):
        from langchain.schema import Document
        self.doc_a = Document(
            page_content="Doc A: Revenue grew significantly.",
            metadata={"chunk_id": "a", "document_id": "d1", "filename": "f.pdf",
                      "page_number": 1, "section": "General"},
        )
        self.doc_b = Document(
            page_content="Doc B: Risk factors mentioned.",
            metadata={"chunk_id": "b", "document_id": "d1", "filename": "f.pdf",
                      "page_number": 2, "section": "Risk Factors"},
        )
        self.doc_c = Document(
            page_content="Doc C: Services segment performance.",
            metadata={"chunk_id": "c", "document_id": "d1", "filename": "f.pdf",
                      "page_number": 3, "section": "Income Statement"},
        )

    def test_rrf_merges_lists(self):
        from app.services.retrieval import reciprocal_rank_fusion
        list1 = [self.doc_a, self.doc_b]
        list2 = [self.doc_b, self.doc_c]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) > 0

    def test_rrf_doc_appearing_in_both_lists_scores_higher(self):
        """doc_b appears in both lists at rank 2 and rank 1 → should score highest."""
        from app.services.retrieval import reciprocal_rank_fusion
        list1 = [self.doc_a, self.doc_b]
        list2 = [self.doc_b, self.doc_c]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        top_chunk_id = fused[0][0].metadata["chunk_id"]
        assert top_chunk_id == "b", "doc_b should rank first as it appears in both lists"

    def test_rrf_scores_are_positive(self):
        from app.services.retrieval import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion([[self.doc_a, self.doc_b]], k=60)
        for _, score in fused:
            assert score > 0


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoint Tests (FastAPI TestClient)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:
    """Integration tests for FastAPI routes using TestClient with mocked services."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch, MagicMock

        with patch("app.services.ingestion.get_vector_store"), \
             patch("app.services.retrieval.get_vector_store"), \
             patch("app.services.rag.get_retrieval_service"):
            from app.main import app
            self.client = TestClient(app)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"] == "FinSight"
        assert "docs" in data

    def test_health_endpoint_structure(self):
        with patch("app.api.health.get_vector_store") as mock_vs, \
             patch("app.api.health.OpenAI") as mock_openai:
            mock_vs.return_value._collection.count.return_value = 42
            mock_openai.return_value.models.list.return_value = []

            resp = self.client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "version" in data
            assert "chroma_status" in data
            assert "openai_status" in data

    def test_query_endpoint_validation_too_short(self):
        resp = self.client.post("/api/v1/query", json={"question": "Hi"})
        assert resp.status_code == 422  # Pydantic validation error

    def test_query_endpoint_valid_request(self):
        from app.models.schemas import QueryResponse, Citation
        from datetime import datetime

        mock_response = QueryResponse(
            question="What was total revenue?",
            answer="Total revenue was $94.9 billion [SOURCE-1].",
            citations=[
                Citation(
                    chunk_id="c1",
                    document_id="d1",
                    filename="apple_10k.pdf",
                    page_number=12,
                    section="Income Statement",
                    relevance_score=0.95,
                    text_snippet="Revenue of $94.9 billion in Q4 2024.",
                )
            ],
            retrieval_latency_ms=42.0,
            generation_latency_ms=850.0,
            total_latency_ms=892.0,
            model_used="gpt-4o",
            tokens_used=320,
        )

        with patch("app.api.query.get_rag_service") as mock_svc:
            mock_svc.return_value.answer.return_value = mock_response
            resp = self.client.post(
                "/api/v1/query",
                json={"question": "What was total revenue in Q4 2024?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "citations" in data
            assert len(data["citations"]) == 1
            assert data["citations"][0]["relevance_score"] == 0.95

