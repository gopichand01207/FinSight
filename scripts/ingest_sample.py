#!/usr/bin/env python3
"""
scripts/ingest_sample.py
────────────────────────
Quick CLI script to ingest a local PDF into the FinSight index.
Used for local testing without the Streamlit UI.

Usage:
    python scripts/ingest_sample.py --file path/to/report.pdf
"""
import argparse
import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into FinSight")
    parser.add_argument("--file", required=True, help="Path to the PDF file")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    if not file_path.suffix.lower() == ".pdf":
        print(f"❌ Only PDF files are supported. Got: {file_path.suffix}")
        sys.exit(1)

    print(f"📄 Ingesting: {file_path.name}")

    from app.services.ingestion import get_ingestion_service
    svc = get_ingestion_service()
    result = svc.ingest(file_path)

    print(f"✅ Ingestion complete!")
    print(f"   Document ID : {result['document_id']}")
    print(f"   Chunks      : {result['num_chunks']}")
    print(f"   Latency     : {result['latency_ms']:.0f} ms")


if __name__ == "__main__":
    main()

