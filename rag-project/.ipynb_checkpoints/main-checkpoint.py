import os
import pickle
import functools
from google.cloud import storage
import numpy as np

import vertexai
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

PROJECT_ID = os.environ.get("GCP_PROJECT", "new-rag-project-prod")
REGION = "europe-west1"
GCS_DATA_BUCKET = f"rag-data-bucket-new-rag-project-prod-{PROJECT_ID}"

@functools.lru_cache(maxsize=1)
def _load_rag_chain():
    print("--- Cold Start: Loading pre-built index data from GCS ---")
    vertexai.init(project=PROJECT_ID, location=REGION)
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_DATA_BUCKET)

    # Download pre-built index files
    bucket.blob("summary_embeddings.pkl").download_to_filename("/tmp/summary_embeddings.pkl")
    bucket.blob("text_summaries.pkl").download_to_filename("/tmp/text_summaries.pkl")
    bucket.blob("original_texts.pkl").download_to_filename("/tmp/original_texts.pkl")
    print("Downloaded index files.")

    # Load the data into memory
    with open("/tmp/summary_embeddings.pkl", "rb") as f:
        summary_embeddings = pickle.load(f)
    with open("/tmp/text_summaries.pkl", "rb") as f:
        text_summaries = pickle.load(f)
    with open("/tmp/original_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    print("Loaded index data into memory.")

    # Reconstruct the SKLearnVectorStore from the loaded data
    embedding_function = VertexAIEmbeddings(model_name="gemini-embedding-001")
    vectorstore = SKLearnVectorStore(
        embedding=embedding_function,
        texts=text_summaries,
        embeddings=summary_embeddings.tolist()
    )
    
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore.as_retriever(), docstore=store)
    retriever.docstore.mset(list(zip([str(i) for i in range(len(texts))], texts)))
    
    model = ChatVertexAI(temperature=0.2, model_name="gemini-1.5-pro-001")
    chain = ({"context": retriever, "question": RunnablePassthrough()}
             | (lambda x: [HumanMessage(content=f"Context: {x['context']}\n\nQuestion: {x['question']}")])
             | model
             | StrOutputParser())
    
    print("--- RAG Chain Initialized ---")
    return chain

import functions_framework
from flask import jsonify

@functions_framework.http
def rag_http_handler(request):
    request_json = request.get_json(silent=True)
    if not request_json or "query" not in request_json:
        return jsonify({"error": "JSON body with a 'query' key is required."}), 400
    query = request_json["query"]
    try:
        rag_chain = _load_rag_chain()
        result = rag_chain.invoke(query)
        return jsonify({"response": result}), 200
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return jsonify({"error": f"Failed to process the request: {str(e)}"}), 500
