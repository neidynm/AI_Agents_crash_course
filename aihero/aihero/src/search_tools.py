import numpy as np
from typing import List, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class VectorSearch:
    """
    Simple vector search implementation using cosine similarity.
    """
    def __init__(self):
        self.embeddings = None
        self.documents = None
    
    def fit(self, embeddings: np.ndarray, documents: List[Dict]):
        """Store embeddings and associated documents."""
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.documents = documents
    
    def search(self, query_embedding: np.ndarray, num_results: int = 5) -> List[Dict]:
        """Search for most similar documents using cosine similarity."""
        if self.embeddings is None:
            return []
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-num_results:][::-1]
        
        # Return documents with similarity scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(similarities[idx])
            results.append(doc)
        
        return results


def create_embeddings(chunks: List[Dict]) -> np.ndarray:
    """Create embeddings for document chunks"""
    evidently_embeddings = []
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    
    print("Encoding document chunks...")
    for d in tqdm(chunks, desc="Creating embeddings"):
        # Create enhanced text for better context
        text_to_encode = d.get('chunk', '')
        
        if 'title' in d and d['title']:
            text_to_encode = f"{d['title']}. {text_to_encode}"
        
        if 'filename' in d:
            filename_parts = d['filename'].replace('/', ' ').replace('_', ' ')
            filename_parts = filename_parts.replace('.mdx', '').replace('.md', '')
            text_to_encode = f"{filename_parts}. {text_to_encode}"
        
        # Encode text to vector
        v = embedding_model.encode(text_to_encode, show_progress_bar=False)
        evidently_embeddings.append(v)
    
    return np.array(evidently_embeddings)


def hybrid_search(
    query: str,
    text_index,
    vector_index: VectorSearch,
    embedding_model: SentenceTransformer,
    alpha: float = 0.5,
    num_results: int = 5
) -> List[Dict]:
    """
    Perform hybrid search combining text and vector search.
    
    Args:
        query: Search query
        text_index: Text search index
        vector_index: Vector search index
        embedding_model: Model to encode query
        alpha: Weight for text search (1-alpha for vector search)
        num_results: Number of results to return
    """
    # Get results from both search methods
    text_results = text_index.search(query, num_results=num_results*2)
    
    q_embedding = embedding_model.encode(query)
    vector_results = vector_index.search(q_embedding, num_results=num_results*2)
    
    # Create a scoring dictionary
    doc_scores = {}
    
    # Add text search scores
    for i, doc in enumerate(text_results):
        doc_key = doc.get('filename', '') + str(doc.get('start', 0))
        text_score = 1.0 / (i + 1)  # Inverse rank
        doc_scores[doc_key] = {
            'doc': doc,
            'text_score': text_score * alpha,
            'vector_score': 0
        }
    
    # Add vector search scores
    for doc in vector_results:
        doc_key = doc.get('filename', '') + str(doc.get('start', 0))
        vector_score = doc.get('similarity_score', 0) * (1 - alpha)
        
        if doc_key in doc_scores:
            doc_scores[doc_key]['vector_score'] = vector_score
        else:
            doc_scores[doc_key] = {
                'doc': doc,
                'text_score': 0,
                'vector_score': vector_score
            }
    
    # Calculate combined scores
    for key in doc_scores:
        doc_scores[key]['combined_score'] = (
            doc_scores[key]['text_score'] + doc_scores[key]['vector_score']
        )
    
    # Sort by combined score and return top results
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    results = [item['doc'] for item in sorted_docs[:num_results]]
    
    return results