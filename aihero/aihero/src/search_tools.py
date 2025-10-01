import io
import zipfile
import requests
import frontmatter

import numpy as np
from typing import List, Any, Dict, Tuple, Optional

from tqdm import tqdm
from minsearch import Index
from sentence_transformers import SentenceTransformer


def read_repo_data(repo_owner, repo_name):
    url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not (filename.endswith('.md') or filename.endswith('.mdx')):
            continue

        with zf.open(file_info) as f_in:
            content = f_in.read()
            post = frontmatter.loads(content)
            data = post.to_dict()

            _, filename_repo = file_info.filename.split('/', maxsplit=1)
            data['filename'] = filename_repo
            repository_data.append(data)

    zf.close()

    return repository_data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size > n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        doc_chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in doc_chunks:
            chunk.update(doc_copy)
        chunks.extend(doc_chunks)

    return chunks


def index_data(
        repo_owner,
        repo_name,
        filter=None,
        chunk=False,
        chunking_params=None,
    ):
    docs = read_repo_data(repo_owner, repo_name)

    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]

    if chunk:
        if chunking_params is None:
            chunking_params = {'size': 2000, 'step': 1000}
        docs = chunk_documents(docs, **chunking_params)

    index = Index(
        text_fields=["content", "filename"],
    )

    index.fit(docs)
    return index

class VectorSearch:
    """
    Simple vector search implementation using cosine similarity.
    """
    def __init__(self):
        self.embeddings = None
        self.documents = None
    
    def fit(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Store embeddings and associated documents.
        """
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
        self.documents = documents
    
    def search(self, query_embedding: np.ndarray, num_results: int = 5) -> List[Dict]:
        """
        Search for most similar documents using cosine similarity.
        """
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

def text_search(query: str) -> List[Any]:
    """
    Perform a text-based search on the FAQ index.
    """
    results = index.search(query, num_results=5)
    print(f"Text search found {len(results)} results for query: '{query}'")
    return results

def vector_search(query: str) -> List[Any]:
    """
    Perform semantic vector-based search.
    """
    # Encode the query into a vector using the embedding model
    q = embedding_model.encode(query)
    # Search the vector index for the top 5 most similar chunks
    results = evidently_vindex.search(q, num_results=5)
    print(f"Vector search found {len(results)} results for query: '{query}'")
    return results

def hybrid_search(query: str, alpha: float = 0.5, num_results: int = 5) -> List[Any]:
    """
    Perform hybrid search combining text and vector search.
    
    Args:
        query: Search query
        alpha: Weight for text search (1-alpha for vector search)
        num_results: Number of results to return
    """
    # Get results from both search methods
    text_results = index.search(query, num_results=num_results*2)
    
    q_embedding = embedding_model.encode(query)
    vector_results = evidently_vindex.search(q_embedding, num_results=num_results*2)
    
    # Create a scoring dictionary
    doc_scores = {}
    
    # Add text search scores
    for i, doc in enumerate(text_results):
        doc_key = doc.get('filename', '') + str(doc.get('start', 0))
        # Use inverse rank as score
        text_score = 1.0 / (i + 1)
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
        doc_scores[key]['combined_score'] = doc_scores[key]['text_score'] + doc_scores[key]['vector_score']
    
    # Sort by combined score and return top results
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x['combined_score'], reverse=True)
    results = [item['doc'] for item in sorted_docs[:num_results]]
    
    print(f"Hybrid search found {len(results)} results for query: '{query}'")
    return results

def create_embeddings (evidently_chunks):
    # Initialize an empty list to store embeddings for each chunk
    evidently_embeddings = []

    # Load a pre-trained sentence transformer model for creating embeddings
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

    # Loop through each document chunk in evidently_chunks
    print("Encoding document chunks...")
    for d in tqdm(evidently_chunks, desc="Creating embeddings"):
        # Create a combined text for better context
        text_to_encode = d['chunk']
        if 'title' in d and d['title']:
            text_to_encode = f"{d['title']}. {text_to_encode}"
        if 'filename' in d:
            # Extract meaningful parts from filename
            filename_parts = d['filename'].replace('/', ' ').replace('_', ' ').replace('.mdx', '').replace('.md', '')
            text_to_encode = f"{filename_parts}. {text_to_encode}"
        
        # Encode the enhanced text into a vector (embedding)
        v = embedding_model.encode(text_to_encode, show_progress_bar=False)
        
        # Append the embedding to the list
        evidently_embeddings.append(v)

    # Convert the list of embeddings into a NumPy array
    return np.array(evidently_embeddings)

