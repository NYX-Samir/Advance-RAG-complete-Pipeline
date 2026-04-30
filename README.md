# Advanced Retrieval-Augmented Generation (RAG) Pipeline

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) system designed with a strong focus on applied LLM systems, retrieval quality, and deployment readiness.

The project implements a modular RAG pipeline, exposes it via a FastAPI service, and supports local LLM inference using Ollama, making it suitable for experimentation, evaluation, and production-oriented learning.

## Key Highlights

* End-to-end RAG pipeline with hybrid retrieval, re-ranking, and context compression
* FastAPI-based backend service for querying and index management
* Dockerized for reproducible local deployment
* Local LLM inference using Ollama, with a backend-agnostic design
* Retrieval evaluation using Recall@K, Precision@K, and MRR
* Modular, extensible codebase suitable for further experimentation

## Architecture Overview

**High-level flow:**

* **Data Ingestion:** Load and preprocess documents from multiple formats.
* **Chunking:** Split documents into manageable, retrieval-friendly chunks.
* **Indexing:** Store embeddings in a vector database (ChromaDB).
* **Retrieval (Hybrid):** Combine dense embedding search with BM25 sparse retrieval.
* **Re-ranking (Optional):** Use cross-encoder models to improve top-K relevance.
* **Context Compression (Optional):** Use an LLM to extract only query-relevant sentences.
* **Generation:** Generate answers using a local LLM backend (Ollama).
* **Evaluation:** Measure retrieval quality and latency trade-offs.

## Features

1.  **Data Ingestion**
    * Supports PDF, TXT, CSV, and HTML documents
    * Metadata preserved for downstream attribution

2.  **Hybrid Retrieval**
    * Dense retrieval using sentence embeddings
    * Sparse retrieval using BM25
    * Improves recall compared to single-retriever setups

3.  **Cross-Encoder Re-Ranking**
    * Re-orders retrieved chunks using cross-encoder models
    * Significantly improves Recall@K and MRR

4.  **Context Compression**
    * LLM-based sentence extraction
    * Reduces prompt size without degrading retrieval quality
    * Includes fallback logic to avoid recall loss

5.  **FastAPI Service**
    * `/query` – query the RAG pipeline
    * `/health` – health check endpoint
    * `/rebuild-index` – rebuild vector index on demand

6.  **Dockerized Deployment**
    * API containerized for reproducible local execution
    * Environment-based configuration via `.env`

## Evaluation Results

The system was evaluated across multiple configurations:

| Configuration | Recall@5 | Precision@5 | MRR | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid + Re-rank + Compression** | 0.90 | 0.36 | 0.867 | ~0.95s |
| **Hybrid (No Re-rank)** | 0.60 | 0.24 | 0.80 | ~0.04s |
| **Hybrid (No Compression)** | 0.90 | 0.36 | 0.867 | ~0.92s |

These results highlight the quality–latency trade-off introduced by re-ranking and compression.

## Project Structure

```plaintext
.
├── app/
│   ├── main.py          # FastAPI application entrypoint
│   ├── schemas.py       # Request / response schemas
│   └── logger.py        # Centralized logging
│
├── src/
│   ├── data_ingestion.py
│   ├── chunking.py
│   ├── vector_embedding.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── context_compression.py
│   ├── evaluation_metrics.py
│   ├── rag_pipeline.py  # Orchestrates the full pipeline
│   └── llm_client.py    # Local LLM client (Ollama)
│
├── UI/
│   └── app.py           # Optional UI entry (demo-oriented)
│
├── data/                # Source documents
├── chroma_db/           # Persisted vector store
├── evaluation/          # Evaluation scripts / artifacts
├── evaluation score .txt
├── Dockerfile.api
├── requirements.txt
├── .env
└── README.md
```

## Running the Project (Local)

**1. Install Dependencies**

```bash
pip install -r requirements.txt

```

**2. Run FastAPI Locally**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

```

**3. Run with Docker**

```bash
docker build -f Dockerfile.api -t rag-api .
docker run -p 8000:8000 rag-api

```

> **Note:** This project uses Ollama for local LLM inference. Ensure Ollama is running and the required model is available.

## Design Notes

* The system is LLM-backend agnostic.
* The local Ollama client can be replaced with a hosted LLM API or inference server in production.
* Re-ranking and context compression are feature-flag controlled to balance quality and latency.
* Designed for single-node, local deployment for learning and evaluation purposes.

## Scope & Intent

This project is intended to:

* Demonstrate applied LLM and RAG system design
* Explore retrieval quality trade-offs
* Practice deployment-ready backend patterns

It is not positioned as a large-scale production system, but as a strong foundation for applied AI engineering.

## Further Reading

For a detailed explanation of the design decisions, trade-offs, and evaluation methodology behind this RAG system, see the technical article:

**Designing a Production-Grade RAG Pipeline From Ingestion to Evaluation**  
https://medium.com/@nyx0samir/designing-a-production-grade-rag-pipeline-from-ingestion-to-evaluation-cea50ff94130

## License

Open for learning, experimentation, and personal projects.

