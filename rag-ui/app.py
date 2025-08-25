import streamlit as st
import requests
import json

# --- Configuration ---
# Paste the Service URL of your deployed Cloud Run API here
API_URL = "https://rag-multimodal-api-693032776487.europe-west1.run.app"

# --- Page Setup ---
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🤖"
)

st.title("🤖 RAG Research Assistant")
st.caption("Ask me questions about the 'Attention Is All You Need' paper.")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Function to Call API ---
def get_rag_response(query):
    """Calls the deployed Cloud Run RAG API."""
    headers = {"Content-Type": "application/json"}
    data = {"query": query}
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=300)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("response", "Sorry, I couldn't get a response.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# --- Chat Interface ---

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_rag_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
