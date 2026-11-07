"""
FinSight Streamlit Dashboard
─────────────────────────────
Multi-page interactive frontend for:
  📂 Document Upload     — drag-and-drop PDF ingestion with progress
  💬 Q&A Chat Interface  — conversational financial document Q&A
  📊 Analytics Panel     — retrieval metrics, citation explorer
  ⚙️  Settings            — configure API endpoint, model, retrieval params

Run with:  streamlit run streamlit_app/main.py
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight — Financial Document Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict]
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_latency_ms" not in st.session_state:
    st.session_state.avg_latency_ms = 0.0

API_BASE = st.session_state.api_base_url


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def api_get(endpoint: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to FinSight API. Is the backend running?")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(endpoint: str, json: dict | None = None, files: dict | None = None) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=json, files=files, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to FinSight API. Is the backend running?")
        return None
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"API error: {detail}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/financial-analytics.png", width=64)
    st.title("FinSight")
    st.caption("AI-Powered Financial Document Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📂 Upload Documents", "💬 Q&A Chat", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.divider()

    # API health badge
    health = api_get("/api/v1/health")
    if health and health.get("status") == "ok":
        st.success("🟢 API Online")
    else:
        st.error("🔴 API Offline")

    st.divider()
    st.caption("FinSight v1.0.0 | RAG + ChromaDB + OpenAI")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UPLOAD DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

if page == "📂 Upload Documents":
    st.header("📂 Upload Financial Documents")
    st.markdown(
        "Upload SEC filings, annual reports, 10-K/10-Q filings, or earnings releases. "
        "FinSight will parse, semantically chunk, and index them for instant Q&A."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Drag and drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Supports SEC filings (10-K, 10-Q, 8-K), annual reports, and earnings releases.",
        )

    with col2:
        st.info(
            "**Supported formats:**\n"
            "- Annual Reports (10-K)\n"
            "- Quarterly Reports (10-Q)\n"
            "- Current Reports (8-K)\n"
            "- Earnings Releases\n"
            "- Prospectuses (S-1)\n"
            "\n**Max file size:** 50 MB"
        )

    if uploaded_files:
        if st.button("🚀 Index Documents", type="primary", use_container_width=True):
            progress = st.progress(0)
            status_area = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_area.info(f"Processing: **{uploaded_file.name}** ({i+1}/{len(uploaded_files)})")
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                result = api_post("/api/v1/documents/upload", files=files)

                if result:
                    st.success(
                        f"✅ **{result['filename']}** — "
                        f"{result['num_chunks']} chunks indexed | "
                        f"ID: `{result['document_id']}`"
                    )

                progress.progress((i + 1) / len(uploaded_files))

            status_area.success("🎉 All documents indexed successfully!")

    # Show indexed documents
    st.subheader("Indexed Documents")
    docs_data = api_get("/api/v1/documents/")
    if docs_data and docs_data.get("documents"):
        import pandas as pd
        df = pd.DataFrame(docs_data["documents"])
        df = df[["filename", "num_chunks", "document_id"]].rename(
            columns={"filename": "Filename", "num_chunks": "Chunks", "document_id": "Document ID"}
        )
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total: {docs_data['total']} documents | "
                   f"{df['Chunks'].sum()} total chunks indexed")
    else:
        st.info("No documents indexed yet. Upload a PDF to get started.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Q&A CHAT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💬 Q&A Chat":
    st.header("💬 Financial Document Q&A")

    # Chat config
    with st.expander("⚙️ Query Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=20, value=5)
            temperature = st.slider("Answer temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        with col2:
            st.markdown("**Restrict to documents:**")
            docs_data = api_get("/api/v1/documents/")
            if docs_data and docs_data.get("documents"):
                doc_options = {d["filename"]: d["document_id"] for d in docs_data["documents"]}
                selected_names = st.multiselect("Select documents (empty = all)", list(doc_options.keys()))
                selected_ids = [doc_options[n] for n in selected_names]
            else:
                selected_ids = []
                st.info("No documents indexed yet.")

    # Sample questions
    with st.expander("💡 Sample Questions", expanded=False):
        samples = [
            "What was the total revenue for the most recent fiscal year?",
            "What are the key risk factors mentioned in the filing?",
            "How did operating income change year over year?",
            "What is the company's current debt-to-equity ratio?",
            "Summarize the management discussion and analysis section.",
            "What were the primary drivers of revenue growth?",
        ]
        for q in samples:
            if st.button(q, use_container_width=True):
                st.session_state.pending_question = q

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📈"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander(f"📎 {len(msg['citations'])} Sources", expanded=False):
                    for idx, cit in enumerate(msg["citations"], 1):
                        st.markdown(
                            f"**[SOURCE-{idx}]** `{cit['filename']}` — "
                            f"Page {cit.get('page_number', '?')} | "
                            f"Section: *{cit.get('section', 'General')}* | "
                            f"Relevance: **{cit['relevance_score']:.2%}**"
                        )
                        st.caption(f"> {cit['text_snippet'][:200]}...")
                        st.divider()

    # Input
    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask anything about your financial documents...") or pending

    if question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        # Query API
        with st.chat_message("assistant", avatar="📈"):
            with st.spinner("Retrieving from financial documents and generating answer..."):
                payload = {
                    "question": question,
                    "top_k": top_k,
                    "temperature": temperature,
                    "document_ids": selected_ids if selected_ids else None,
                }
                result = api_post("/api/v1/query", json=payload)

            if result:
                st.markdown(result["answer"])

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Latency", f"{result['total_latency_ms']:.0f} ms")
                col2.metric("Retrieval", f"{result['retrieval_latency_ms']:.0f} ms")
                col3.metric("Generation", f"{result['generation_latency_ms']:.0f} ms")
                col4.metric("Citations", len(result.get("citations", [])))

                # Update session analytics
                st.session_state.total_queries += 1
                prev_avg = st.session_state.avg_latency_ms
                n = st.session_state.total_queries
                st.session_state.avg_latency_ms = prev_avg + (result["total_latency_ms"] - prev_avg) / n

                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                    "metrics": {
                        "total_ms": result["total_latency_ms"],
                        "retrieval_ms": result["retrieval_latency_ms"],
                        "generation_ms": result["generation_latency_ms"],
                    },
                })

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Analytics":
    st.header("📊 System Analytics")

    health = api_get("/api/v1/health")
    docs_data = api_get("/api/v1/documents/")

    total_docs = docs_data["total"] if docs_data else 0
    total_chunks = sum(d["num_chunks"] for d in docs_data["documents"]) if docs_data else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Documents Indexed", total_docs)
    col2.metric("🧩 Total Chunks", total_chunks)
    col3.metric("💬 Queries Run", st.session_state.total_queries)
    col4.metric("⚡ Avg Latency", f"{st.session_state.avg_latency_ms:.0f} ms")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("System Health")
        if health:
            st.json({
                "API Status": health.get("status"),
                "Version": health.get("version"),
                "ChromaDB": health.get("chroma_status"),
                "OpenAI": health.get("openai_status"),
            })

    with col_r:
        st.subheader("Pipeline Architecture")
        st.markdown("""
```
 PDF Upload
     │
     ▼
 PyMuPDF Parser ──► Page-level text + metadata
     │
     ▼
 SemanticChunker ──► Sentence-aware chunks (100–512 tokens)
     │               with 50-token overlap
     ▼
 OpenAI Embeddings ──► text-embedding-3-large (3072-dim)
     │
     ▼
 ChromaDB ──────────► Persistent vector index
     │
     ▼
 Hybrid Retrieval:
   ├─ Dense  ──► Cosine similarity (ChromaDB)
   └─ Sparse ──► BM25 (rank_bm25)
         │
         ▼
 Reciprocal Rank Fusion (RRF k=60)
         │
         ▼
 GPT-4o Generation with citations
         │
         ▼
 Structured QueryResponse (JSON)
```
        """)

    # Query latency history (from chat history)
    if st.session_state.chat_history:
        latencies = [
            msg["metrics"]["total_ms"]
            for msg in st.session_state.chat_history
            if msg["role"] == "assistant" and "metrics" in msg
        ]
        if latencies:
            import pandas as pd
            st.subheader("Query Latency History")
            df_lat = pd.DataFrame({"Query #": range(1, len(latencies)+1), "Latency (ms)": latencies})
            st.line_chart(df_lat.set_index("Query #"))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")

    st.subheader("API Configuration")
    new_api_url = st.text_input(
        "FastAPI Backend URL",
        value=st.session_state.api_base_url,
        help="URL of the FinSight FastAPI backend",
    )
    if st.button("Save API URL"):
        st.session_state.api_base_url = new_api_url
        st.success(f"API URL updated to: {new_api_url}")

    st.divider()
    st.subheader("About FinSight")
    st.markdown("""
| Component | Technology |
|-----------|-----------|
| **LLM** | OpenAI GPT-4o |
| **Embeddings** | text-embedding-3-large (3072-dim) |
| **Vector DB** | ChromaDB (persistent) |
| **RAG Framework** | LangChain LCEL |
| **Retrieval** | Hybrid Dense + BM25 → RRF |
| **Backend** | FastAPI (async) |
| **Frontend** | Streamlit |
| **Deployment** | AWS EC2 + Docker |
| **CI/CD** | GitHub Actions |
    """)

