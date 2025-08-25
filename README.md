# RAG Research Assistant on Google Cloud

**A complete, end-to-end RAG application that serves as an expert on the "Attention Is All You Need" paper, featuring a scalable API on Cloud Run and an interactive Streamlit UI.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Introduction

### 1.1. Purpose & Objective
This project demonstrates the creation of a sophisticated AI research assistant using a Retrieval-Augmented Generation (RAG) architecture. The primary objective is to build a system that can answer complex questions about a specific, dense technical document—the seminal AI paper, "Attention Is All You Need."

The goal is to overcome the limitations of standard Large Language Models (LLMs) by grounding their responses in factual, verifiable information extracted directly from the source material.

### 1.2. Goals
-   **Develop a RAG Pipeline:** Create a system that can ingest, process, and retrieve information from a PDF document.
-   **Achieve Production-Ready Architecture:** Implement a scalable, two-part architecture that separates the heavy data-processing (indexing) from the lightweight application serving.
-   **Deploy a Serverless API:** Host the RAG agent as a scalable, public API using Google Cloud Run.
-   **Create an Interactive Frontend:** Build a user-friendly chatbot interface with Streamlit to interact with the deployed API.

## 2. Development & Methodology

The project was executed in two distinct phases, following a professional MLOps workflow.

### 2.1. Phase 1: The Indexing Job
The first phase involved creating the RAG agent's "brain." This is a one-time, offline process designed to be run in a powerful development environment like Vertex AI Workbench.

**Process:**
1.  **Data Ingestion:** The "Attention Is All You Need" PDF was downloaded from arXiv.
2.  **Text Extraction & Summarization:** The text from each page was extracted. A Gemini model (`gemini-2.5-flash`) was used to generate a concise, semantically rich summary for each page.
3.  **Vector Embedding:** The summaries were converted into numerical representations (vector embeddings) using a Vertex AI Embedding model (`gemini-embedding-001`).
4.  **Index Persistence:** The embeddings and the original texts were saved to separate `.pkl` files. This separation of data from live model objects is crucial for creating a portable and serializable index.
5.  **Cloud Storage:** The final index files were uploaded to a Google Cloud Storage bucket to be accessed by the serving application.

### 2.2. Phase 2: The Serving Application
The second phase involved creating and deploying a lightweight, fast-starting server to handle user queries.

**Process:**
1.  **Containerization:** A `Dockerfile` was created to define a clean, reproducible Python environment, ensuring the application runs consistently.
2.  **API Server (`main.py`):** A Python application was built using the `functions-framework`. On its first startup (cold start), it:
    -   Downloads the pre-built index files from the GCS bucket.
    -   Loads the index data into memory.
    -   Reconstructs the `SKLearnVectorStore`.
    -   Initializes the final LangChain RAG pipeline.
3.  **Deployment:** The containerized application was deployed as a serverless API to **Google Cloud Run**. This provides automatic scaling (including scaling to zero to save costs), high availability, and a public HTTPS endpoint.


### 2.3. Technologies, Tools, and APIs Used

-   **Cloud Platform:** Google Cloud Platform (GCP)
-   **AI Models & APIs:**
    -   **Vertex AI:** `gemini-2.5-pro`, `gemini-2.5-flash` (for summarization and answer generation), `gemini-embedding-001` (for vector embeddings).
-   **Orchestration Framework:** LangChain
-   **Vector Store:** `scikit-learn` (`SKLearnVectorStore`) for an efficient in-memory index.
-   **Deployment & Serving:**
    -   **Cloud Run:** For deploying the scalable, serverless API.
    -   **Docker:** For containerizing the application.
-   **Development Environment:** Vertex AI Workbench
-   **Frontend UI:** Streamlit
-   **Storage:** Google Cloud Storage (GCS)
-   **Programming Language:** Python 3.12

## 3. Conclusion

This project successfully achieved its goal of building a complete, end-to-end RAG application. We systematically progressed from initial concept and debugging in a notebook environment to a robust, two-part production architecture deployed on Google Cloud.

The final system demonstrates a powerful and scalable pattern for creating specialized AI assistants. The separation of the indexing job from the serving application proved to be the critical architectural decision that solved the "cold start" and dependency issues inherent in serverless deployments. The result is a fast, reliable, and intelligent research assistant, accessible via a public API and an interactive web interface.
