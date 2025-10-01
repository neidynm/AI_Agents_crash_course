import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import sys
from pathlib import Path

# Load environment variables
load_dotenv()

# Add src directory to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from ingest import ingest_repo, build_text_index
from search_tools import create_embeddings, VectorSearch, hybrid_search
import numpy as np
from sentence_transformers import SentenceTransformer


# --- Configure Streamlit ---
st.set_page_config(page_title="FAQ Agent", page_icon="🤖", layout="centered")
st.title("🤖 FAQ Agent - Evidently AI Docs")

# --- Initialize OpenRouter Client ---
@st.cache_resource
def init_openrouter():
    """Initialize OpenRouter client"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("⚠️ OPENROUTER_API_KEY not found in environment variables!")
        st.stop()
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

openai_client = init_openrouter()

# --- Initialize Search Indices ---
@st.cache_resource
def initialize_search():
    """Initialize text and vector search indices"""
    with st.spinner("📥 Loading Evidently documentation..."):
        # Ingest documents
        chunks = ingest_repo("evidentlyai", "docs")
        st.success(f"✅ Loaded {len(chunks)} document chunks")
        
        # Build text index
        with st.spinner("🔍 Building text search index..."):
            text_index = build_text_index(chunks)
        
        # Build vector index
        with st.spinner("🧮 Creating embeddings (this may take a few minutes)..."):
            embeddings = create_embeddings(chunks)
            vector_index = VectorSearch()
            vector_index.fit(embeddings, chunks)
            
        # Load embedding model for queries
        embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
        
        st.success("✅ Search system initialized!")
        
    return text_index, vector_index, embedding_model, chunks

# Initialize (only runs once due to caching)
try:
    text_index, vector_index, embedding_model, chunks = initialize_search()
except Exception as e:
    st.error(f"❌ Error initializing search: {e}")
    st.stop()

# --- Search Function ---
def search_docs(query, search_method='hybrid', num_results=5):
    """Search documentation using specified method"""
    try:
        if search_method == 'text':
            results = text_index.search(query, num_results=num_results)
        elif search_method == 'vector':
            q_embedding = embedding_model.encode(query)
            results = vector_index.search(q_embedding, num_results=num_results)
        else:  # hybrid
            results = hybrid_search(
                query=query,
                text_index=text_index,
                vector_index=vector_index,
                embedding_model=embedding_model,
                num_results=num_results
            )
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# --- Answer Question Function ---
def answer_question(question, search_method='hybrid'):
    """Answer question using search + LLM"""
    # Search for relevant docs
    search_results = search_docs(question, search_method=search_method, num_results=5)
    
    if not search_results:
        return "I couldn't find any relevant information in the documentation."
    
    # Format context
    context = "\n\n---\n\n".join([
        f"Source: {result.get('filename', 'unknown')}\n{result.get('chunk', '')}"
        for result in search_results
    ])
    
    # Create prompt
    prompt = f"""You are an expert assistant that answers questions about the Evidently project 
(https://github.com/evidentlyai/evidently) using ONLY the information provided below.

Context from Evidently documentation:
{context}

User question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be concise and clear
- If the answer is not in the context, say "I could not find this information in the Evidently documentation"
- Do not invent features or functionality

Answer:"""
    
    try:
        response = openai_client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about Evidently."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting answer: {e}"

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    search_method = st.selectbox(
        "Search Method",
        options=['hybrid', 'text', 'vector'],
        help="Choose how to search the documentation"
    )
    
    st.divider()
    st.markdown("""
    ### About
    This app answers questions about **Evidently AI** using:
    - 📚 Evidently documentation
    - 🔍 Hybrid search (text + semantic)
    - 🤖 DeepSeek LLM via OpenRouter
    """)

# --- Initialize session state for chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask me anything about Evidently AI..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = answer_question(prompt, search_method=search_method)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Example questions ---
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    example_questions = [
        "What components are required in a test dataset to evaluate AI?",
        "How to run evaluations in Evidently?",
        "How to create monitoring dashboards?",
        "What is data drift detection?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}"):
                st.rerun()