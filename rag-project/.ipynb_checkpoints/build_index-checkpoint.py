import pickle
import requests
import vertexai
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI

print("--- Starting Index Build Job ---")
vertexai.init(project="new-rag-project-prod", location="europe-west1")

texts = [d.page_content for d in PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf").load() if d.page_content.strip()]
print(f"Loaded {len(texts)} pages.")

# --- MODEL NAME CORRECTED ---
summarizer = VertexAI(model_name="gemini-2.5-flash")
summaries = summarizer.batch(texts)
print(f"Generated {len(summaries)} summaries.")

# --- MODEL NAME CORRECTED ---
embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001").embed_documents(summaries)
print("Created embeddings.")

with open("summary_embeddings.pkl", "wb") as f:
    pickle.dump(np.array(embeddings), f)
with open("original_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("--- Index Build Job Complete ---")
