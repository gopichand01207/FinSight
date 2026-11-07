"""
FinSight Configuration Module
Centralized settings management using Pydantic BaseSettings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    Uses Pydantic v2 BaseSettings for type-safe configuration.
    """

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o",
        description="LLM model for generation (gpt-4o / gpt-4-turbo)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model for dense retrieval",
    )
    embedding_dimensions: int = Field(
        default=3072,
        description="Dimensionality of text-embedding-3-large vectors",
    )

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Local directory where ChromaDB persists vector data",
    )
    chroma_collection_name: str = Field(
        default="finsight_financial_docs",
        description="ChromaDB collection name",
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    # Dense retrieval
    dense_top_k: int = Field(
        default=10,
        description="Number of dense (embedding) candidates to retrieve",
    )
    # Sparse / BM25 retrieval
    sparse_top_k: int = Field(
        default=10,
        description="Number of sparse (BM25 keyword) candidates to retrieve",
    )
    # Hybrid fusion final result count
    hybrid_top_k: int = Field(
        default=5,
        description="Final number of chunks after Reciprocal Rank Fusion",
    )
    rrf_k: int = Field(
        default=60,
        description="Constant k used in Reciprocal Rank Fusion (default=60)",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    semantic_chunk_min_tokens: int = Field(
        default=100,
        description="Minimum token count per semantic chunk",
    )
    semantic_chunk_max_tokens: int = Field(
        default=512,
        description="Maximum token count per semantic chunk",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        description="Overlap in tokens between consecutive chunks",
    )

    # ── FastAPI ──────────────────────────────────────────────────────────────
    app_name: str = "FinSight"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    allowed_origins: list[str] = Field(
        default=["http://localhost:8501"],
        description="CORS allowed origins (Streamlit default port)",
    )

    # ── Storage ──────────────────────────────────────────────────────────────
    upload_dir: str = Field(
        default="./data/uploads",
        description="Temporary directory for uploaded financial documents",
    )
    max_upload_size_mb: int = Field(
        default=50,
        description="Max allowed upload file size in MB",
    )

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_region: Optional[str] = Field(default=None)
    s3_bucket_name: Optional[str] = Field(
        default=None,
        description="Optional S3 bucket for persistent document storage",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — import this throughout the app
settings = Settings()


def get_settings() -> Settings:
    """FastAPI dependency injection helper."""
    return settings

