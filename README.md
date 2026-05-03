# Advanced Retrieval-Augmented Generation (RAG) Pipeline

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) system designed with a strong focus on applied LLM systems, retrieval quality, microservices architecture, and production-ready cloud deployment.

The project implements a modular RAG pipeline, exposes it via a FastAPI backend, serves an interactive Streamlit frontend, and utilizes the Groq API for ultra-low latency inference. The entire system is containerized using Docker and deployed on Hugging Face Spaces.

## Live Demo
* **Interactive UI (Streamlit):** https://huggingface.co/spaces/nyx-samir/rag-streamlit-ui
* **Backend API (FastAPI):** https://huggingface.co/spaces/nyx-samir/rag-fastapi-backend

## Key Highlights

* **Cloud-Native Architecture:** Deployed as independent microservices (Backend + Frontend) on Hugging Face Spaces using Docker.
* **Ultra-Fast Generation:** Integrated with **Groq API** (Llama 3 / Mixtral) replacing local Ollama for production-grade speed.
* **Advanced Retrieval:** Hybrid retrieval (Dense + BM25), Cross-Encoder Re-ranking, and LLM-based Context Compression.
* **Container Orchestration:** Fully reproducible local environment using `docker-compose`.
* **Large File Management:** Vector databases (ChromaDB) and source PDFs tracked and deployed via **Git LFS**.

## Architecture Overview

**High-level flow:**
1. **Data Ingestion & Chunking:** Load PDFs/documents and split them into retrieval-friendly chunks.
2. **Indexing:** Store embeddings in a persisted vector database (ChromaDB).
3. **Hybrid Retrieval:** Combine dense embedding search with BM25 sparse retrieval.
4. **Re-ranking:** Use cross-encoder models to re-order and improve top-K relevance.
5. **Context Compression:** Extract only query-relevant sentences to optimize prompt context window.
6. **Generation:** Generate high-quality answers using **Groq API** for near-instant inference.
7. **Microservices Communication:** Streamlit UI communicates seamlessly with the FastAPI backend over REST.

## Features

1. **Hybrid Retrieval & Cross-Encoder Re-Ranking**
   * Dense (Embeddings) + Sparse (BM25) retrieval improves base recall.
   * Cross-encoder significantly improves Recall@K and MRR.
2. **Context Compression**
   * Reduces prompt size and token cost without degrading retrieval quality.
3. **Microservices API & UI**
   * **FastAPI:** Core backend handling LLM logic and ChromaDB operations.
   * **Streamlit:** Clean, interactive, chat-like frontend with latency tracking and health checks.
4. **Dockerized Orchestration**
   * `docker-compose.yml` setup for 1-click local testing.
   * Individual `Dockerfile` setups optimized for Hugging Face Spaces (Port 7860).

## Evaluation Results

The system was rigorously evaluated on a custom dataset. The results highlight the significant quality-latency trade-offs introduced by architectural choices:

### 1. Retrieval Performance
| Configuration | Recall@5 | Precision@5 | MRR | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid + Re-rank** | **0.800** | **0.320** | **0.650** | ~0.95s |
| **Hybrid (No Re-rank)** | 0.300 | 0.120 | 0.440 | **~0.08s** |

*Insight: Integrating a Cross-Encoder for re-ranking yielded a ~2.6x improvement in Recall@5, validating the necessity of re-ranking for complex policy documents, despite a ~870ms latency penalty.*

### 2. Context Compression & Cost Efficiency
To optimize LLM prompt context and reduce token expenditure, an LLM-based context compressor was utilized before generation.

* **Original Context Size:** 18,413 characters
* **Compressed Context Size:** 3,826 characters
* **Tokens Saved per Query:** ~3,646 tokens
* **Overall Cost Reduction:** **79.22%**

*Insight: The compression step reduced API token payload by nearly 80%, drastically cutting down potential inference costs while maintaining high answer relevance. The generation time saw a minimal shift (1.02s without compression vs. 1.24s with compression) due to the preprocessing overhead.*
## Project Structure

```plaintext
.
├── app/
│   ├── main.py          # FastAPI application entrypoint
│   ├── app.py           # Streamlit UI frontend
│   ├── schemas.py       # Request / response schemas
│   └── logger.py        # Centralized logging
│
├── src/                 # Core RAG logic (chunking, embedding, retrieval, reranking)
├── data/                # Source documents (Tracked via Git LFS)
├── chroma_db/           # Persisted vector store (Tracked via Git LFS)
├── docker-compose.yml   # Multi-container orchestration
├── Dockerfile.api       # Backend Docker image (Exposed on 8000 local / 7860 cloud)
├── Dockerfile.ui        # Frontend Docker image (Exposed on 8501 local / 7860 cloud)
├── requirements.txt     # Backend dependencies
├── requirements.ui.txt  # Frontend dependencies
├── .env.example         # Environment variable templates
├── .gitattributes       # Git LFS configuration
└── README.md
```

## Running the Project (Local Testing)

**1. Clone and Setup Environment**
```bash
git clone [https://github.com/NYX-Samir/Advance-RAG-complete-Pipeline](https://github.com/NYX-Samir/Advance-RAG-complete-Pipeline)
cd Advance-RAG-complete-Pipeline
```

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
ENABLE_RERANK=true
ENABLE_COMPRESSION=true
DATA_PATH=./data
CHROMA_DIR=./chroma_db
```

**2. Launch with Docker Compose**
```bash
docker-compose up --build -d
```
* **Frontend UI:** Available at `http://localhost:8501`
* **Backend API:** Available at `http://localhost:8000/docs`

## Cloud Deployment (Hugging Face Spaces)

This project is deployed using two separate Hugging Face Docker Spaces:
1. **Backend Space:** Runs `Dockerfile.api` (Port mapped to 7860). Employs **Git LFS** to push `chroma.sqlite3` and large PDFs.
2. **Frontend Space:** Runs `Dockerfile.ui` (Port mapped to 7860) and points `API_URL` to the remote backend Space.

## Further Reading

For a detailed explanation of the design decisions, trade-offs, and evaluation methodology behind the core RAG logic, see my technical article:

**Designing a Production-Grade RAG Pipeline From Ingestion to Evaluation** [Read on Medium](https://medium.com/@nyx0samir/designing-a-production-grade-rag-pipeline-from-ingestion-to-evaluation-cea50ff94130)

## License

Open for learning, experimentation, and personal projects.
```
