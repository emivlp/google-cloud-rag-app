# Multimodal RAG Research Assistant on Google Cloud

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete source code for an advanced, multimodal Retrieval-Augmented Generation (RAG) application. The system functions as a research assistant capable of answering complex questions about the seminal AI paper, **"Attention Is All You Need,"** by analyzing both its text and architectural diagrams.

The project is built with Google's **Gemini** models, orchestrated with **LangChain**, and deployed as a scalable, serverless API on **Google Cloud Run**, complete with an interactive UI built with **Streamlit**.



## 📋 Table of Contents
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [API Usage](#-api-usage)

---
## ✨ Features

-   **Multimodal Understanding:** Ingests and processes both text from a PDF and information from diagrams.
-   **Advanced RAG Pipeline:** Uses a Multi-Vector Retriever strategy with an in-memory FAISS index for efficient and accurate context retrieval.
-   **Serverless & Scalable:** The backend API is deployed on Google Cloud Run, allowing it to scale automatically from zero to handle high traffic loads.
-   **Interactive UI:** A user-friendly chatbot interface built with Streamlit for easy interaction with the RAG agent.
-   **Separation of Concerns:** A professional two-step architecture separates the one-time, heavy indexing job from the lightweight, fast-starting serving application.

---
## 🏛️ Architecture

This project follows a standard production pattern for ML systems, separating the data processing (indexing) from the application serving.

### 1. The Indexing Job (`rag-project`)
A one-time Python script that performs the heavy lifting:
1.  Downloads the source PDF ("Attention Is All You Need").
2.  Extracts the text content from each page.
3.  Uses a Gemini model (`gemini-1.5-flash-001`) to generate a concise summary for each page.
4.  Uses a Vertex AI embedding model (`text-embedding-004`) to convert these summaries into vector embeddings.
5.  Constructs a `scikit-learn` vector store from these embeddings.
6.  Saves the final, queryable index and the original texts to `.pkl` files.

### 2. The Serving Application (`rag-server`)
A lightweight, containerized web service deployed on Cloud Run:
1.  **On Cold Start:** Downloads the pre-built index files (`.pkl`) from a Google Cloud Storage (GCS) bucket.
2.  **Initializes a RAG Chain:** Loads the index into memory and constructs a LangChain RAG pipeline.
3.  **Serves Requests:** Exposes an HTTP endpoint that accepts a user's query, passes it to the RAG chain, and returns the model's generated answer.



---
## 📁 Project Structure

The repository is organized into two main folders, reflecting the two-part architecture.

```
.
├── .gitignore
├── rag-project/
│   ├── build_index.py        # Script to create the vector store
│   └── requirements.txt      # Dependencies for the build script
└── rag-ui/
    ├── app.py                # The Streamlit UI application code
    └── requirements.txt      # Dependencies for the Streamlit app
```

*(Note: The `rag-server` deployment files were created in the final steps but are consolidated here for clarity in a real-world repository.)*

---
## 🚀 Getting Started

To run this project, you'll need a Google Cloud project with billing enabled and the necessary APIs activated (`Vertex AI`, `Cloud Run`, `Cloud Build`, `Cloud Storage`).

### 1. Set Up the Environment
It is highly recommended to use a **Vertex AI Workbench** instance as the development environment to avoid dependency and resource issues.

### 2. Build the Index
This is a one-time process to create your RAG agent's "brain."

```bash
# Navigate to the indexing directory
cd rag-project

# Install dependencies
pip install -r requirements.txt

# Run the build script
python build_index.py

# Create a GCS bucket and upload the index
# (Replace with your project ID)
GCS_BUCKET="gs://rag-data-bucket-your-project-id"
gcloud storage buckets create $GCS_BUCKET --location=europe-west1
gcloud storage cp *.pkl $GCS_BUCKET/
```

---
## ☁️ Deployment

Once the index is built and stored in GCS, you can deploy the serving application.

### Deploy the RAG API to Cloud Run

The serving application is defined in the `rag-server` directory (which you created during the tutorial). It contains `main.py`, `Dockerfile`, and its own `requirements.txt`.

From within that directory, run:
```bash
gcloud run deploy rag-multimodal-api \
    --source . \
    --platform managed \
    --region europe-west1 \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --timeout=300s \
    --clear-base-image
```

After deployment, make the service public by running the command suggested in the output or by using the Cloud Console.

### Run the Streamlit UI
From the `rag-ui` directory:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (replace the API_URL in app.py with your deployed URL)
streamlit run app.py
```

---
## 📞 API Usage

You can query the deployed API endpoint directly using `curl`.

```bash
# Replace with your actual Cloud Run Service URL
SERVICE_URL="[https://rag-multimodal-api-....run.app](https://rag-multimodal-api-....run.app)"

curl -X POST $SERVICE_URL \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the purpose of multi-head attention?"}'
```

**Example Response:**
```json
{
  "response": "Based on the context provided, multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions..."
}
```
