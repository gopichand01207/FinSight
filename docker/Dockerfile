# ──────────────────────────────────────────────────────────────────
# FinSight FastAPI Backend — Multi-stage Docker Build
# ──────────────────────────────────────────────────────────────────
# Stage 1: Builder  — install dependencies in a venv
# Stage 2: Runtime  — slim production image (~200 MB)
# ──────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="FinSight API"
LABEL org.opencontainers.image.description="AI-Powered Financial Document Intelligence"
LABEL org.opencontainers.image.version="1.0.0"
LABEL maintainer="Bhukya Gopi Chand Naik"

# Runtime system deps (PyMuPDF needs libmupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r finsight && useradd -r -g finsight -d /app finsight

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY streamlit_app/ ./streamlit_app/

# Create required directories
RUN mkdir -p data/uploads chroma_db && \
    chown -R finsight:finsight /app

USER finsight

# Build metadata args
ARG BUILD_DATE
ARG GIT_SHA
ENV BUILD_DATE=${BUILD_DATE} GIT_SHA=${GIT_SHA}

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log"]

