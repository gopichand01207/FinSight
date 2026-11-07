#!/usr/bin/env python3
"""
scripts/query_cli.py
────────────────────
Interactive command-line Q&A interface for FinSight.
Useful for quick testing without starting the full Streamlit UI.

Usage:
    python scripts/query_cli.py
    python scripts/query_cli.py --question "What was total revenue?"
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_query(question: str, top_k: int = 5) -> None:
    from app.services.rag import get_rag_service
    from app.models.schemas import QueryRequest

    print(f"\n🔍 Query: {question}")
    print("⏳ Running hybrid retrieval + generation...\n")

    svc = get_rag_service()
    request = QueryRequest(question=question, top_k=top_k, temperature=0.0)
    response = svc.answer(request)

    print("=" * 70)
    print(f"📝 ANSWER\n{response.answer}")
    print("=" * 70)
    print(f"\n📎 SOURCES ({len(response.citations)} citations)")
    for i, cit in enumerate(response.citations, 1):
        print(f"  [{i}] {cit.filename} | Page {cit.page_number} | "
              f"Section: {cit.section} | Score: {cit.relevance_score:.2%}")
        print(f"      \"{cit.text_snippet[:120]}...\"")

    print(f"\n⚡ Latency: {response.total_latency_ms:.0f} ms "
          f"(retrieval: {response.retrieval_latency_ms:.0f} ms | "
          f"generation: {response.generation_latency_ms:.0f} ms)")


def interactive_mode(top_k: int = 5) -> None:
    print("\n🌟 FinSight Interactive Q&A CLI")
    print("   Type your question and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Exiting.")
            break
        if len(question) < 5:
            print("⚠️  Please enter a longer question.")
            continue
        run_query(question, top_k=top_k)
        print()


def main():
    parser = argparse.ArgumentParser(description="FinSight CLI Query Tool")
    parser.add_argument("--question", "-q", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.question:
        run_query(args.question, top_k=args.top_k)
    else:
        interactive_mode(top_k=args.top_k)


if __name__ == "__main__":
    main()

