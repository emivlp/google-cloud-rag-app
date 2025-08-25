import os
import pickle
import functools
from google.cloud import storage
import numpy as np
import vertexai
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

PROJECT_ID = os.environ.get("GCP_PROJECT", "new-rag-project-prod")
REGION = "europe-west1"
GCS_DATA_BUCKET = f"rag-data-bucket-{PROJECT_ID}"

@functools.lru_cache(maxsize=1)
def _load_rag_chain():
    print("--- Cold Start: Loading pre-built index from GCS ---")
    vertexai.init(project=PROJECT_ID, location=REGION)
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_DATA_BUCKET)

    blob_embeddings = bucket.blob("summary_embeddings.pkl")
    blob_embeddings.download_to_filename("/tmp/summary_embeddings.pkl")
    blob_texts = bucket.blob("original_texts.pkl")
    blob_texts.download_to_filename("/tmp/original_texts.pkl")
    print("Downloaded index files.")

    with open("/tmp/summary_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open("/tmp/original_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    print("Loaded index data into memory.")

    # --- MODEL NAMES CORRECTED ---
    embedding_function = VertexAIEmbeddings(model_name="gemini-embedding-001", location=REGION)
    model = ChatVertexAI(model_name="gemini-2.5-pro", temperature=0.2, location=REGION)
    
    vectorstore = SKLearnVectorStore(embedding=embedding_function)
    vectorstore.add_texts(texts=texts, embeddings=embeddings.tolist())
    
    retriever = vectorstore.as_retriever()
    
    template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
    prompt = PromptTemplate.from_template(template)

    chain = (
        RunnableParallel(context=retriever, question=RunnablePassthrough())
        | prompt
        | model
        | StrOutputParser()
    )
    
    print("--- RAG Chain Initialized Successfully ---")
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
