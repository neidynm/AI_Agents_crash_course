import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from search_tools import hybrid_search

# Setup OpenAI client for OpenRouter
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Load embedding model (used for vector and hybrid search)
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')


def answer_question_manual(question, search_method, index, vindex):
    """
    Answer a question using search + LLM.
    
    Args:
        question: The user's question
        search_method: 'text', 'vector', or 'hybrid'
        index: Text search index (minsearch.Index)
        vindex: Vector search index (VectorSearch)
    
    Returns:
        str: The answer from the LLM
    """
    # Perform search based on method
    if search_method == "text":
        search_results = index.search(question, num_results=5)
        print(f"🔍 Found {len(search_results)} results using text search")
    elif search_method == "vector":
        q_embedding = embedding_model.encode(question)
        search_results = vindex.search(q_embedding, num_results=5)
        print(f"🔍 Found {len(search_results)} results using vector search")
    else:  # hybrid
        search_results = hybrid_search(
            query=question,
            text_index=index,
            vector_index=vindex,
            embedding_model=embedding_model,
            alpha=0.5,
            num_results=5
        )
        print(f"🔍 Found {len(search_results)} results using hybrid search")
    
    if not search_results:
        return "I couldn't find any relevant information in the Evidently documentation."
    
    # Format context from search results
    context = "\n\n---\n\n".join([
        f"Source: {result.get('filename', 'unknown')}\n{result.get('chunk', '')}"
        for result in search_results
    ])
    
    # Build prompt
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
        print("🤖 Generating answer...")
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
        return f"Error getting answer from LLM: {e}\n\nPlease check your OPENROUTER_API_KEY in .env file"