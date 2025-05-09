# Agent Maker Project

This project consists of an advanced FastAPI backend and a Next.js frontend for orchestrating AI agents and implementing Retrieval Augmented Generation (RAG) pipelines.

## Project Overview

**Backend (FastAPI - `agent_maker` directory):**

*   **Model Management:** Supports NVIDIA NIMs for chat and embeddings. Endpoints to list models and generate embeddings.
*   **Prompt Library:** Manages prompt templates (JSON/YAML) with CRUD-like operations and Pydantic validation.
*   **Tool Library:** 
    *   Lists and provides details for curated LangChain tools.
    *   Placeholder for custom/MCP tool integration.
    *   Basic execution endpoints for LangChain and MCP tools with logging.
*   **Agent Orchestration (LangGraph):**
    *   Manages agent definitions (stored as JSON).
    *   Supports `react` and `tool-calling` agent types.
    *   Integrates with model, prompt, and tool libraries.
    *   Offers optional conversation memory.
    *   Streams agent responses via Server-Sent Events (SSE).
*   **Advanced RAG Pipelines:**
    *   **Document Handling:** Upload, auto-detect loader, chunking (recursive, token-based etc.) via a modular `Chunker` class.
    *   **Vector Store Management:** Manages FAISS (with persistence) and ChromaDB vector stores using NVIDIA embeddings through a `VectorStoreManager`.
    *   **Indexing:** Endpoint to index document chunks.
    *   **Search & Retrieval:**
        *   Basic vector search and hybrid search capabilities.
        *   Advanced retrievers: `MultiQueryRetriever`, `EnsembleRetriever` (with placeholder keyword retriever for FAISS), and `LongContextReorder`.
        *   **Reranking:** Integrated `NVIDIARerank` for improving search result relevance.
    *   **Vector Store Operations:** Delete/update chunks (currently ChromaDB only).

**Frontend (Next.js - `frontend` directory):**

*   Initialized Next.js project with TypeScript, Tailwind CSS, and App Router.
*   Basic page to list available AI agents fetched from the backend (`/agents`).

**Testing & Quality:**

*   Pytest for backend API testing (`agent_maker/tests/test_api.py`).
*   Ruff for linting and formatting.

## Current Status

The FastAPI backend has extensive functionality. Core testing and linting issues have been addressed. The Next.js frontend project has been initialized, and the backend is configured for CORS to allow frontend communication.

## Next Steps (Examples)

*   Address remaining deprecation warnings in the backend.
*   Fix the skipped streaming agent test.
*   Implement a robust keyword retriever for FAISS in the RAG pipeline.
*   Fully implement custom/MCP tool integration and execution.
*   Continue frontend development (e.g., agent interaction UI, RAG pipeline UI).

## Setup & Running

(Instructions to be added here once the repository is set up, e.g., cloning, environment setup, running backend and frontend servers.) 