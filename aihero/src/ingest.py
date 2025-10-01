import io
import os
import zipfile
import requests
import frontmatter
import logging
from tqdm import tqdm
from minsearch import Index


def read_repo_data(repo_owner, repo_name, branch="main"):
    """
    Download and parse all markdown files from a GitHub repository.
    Yields one document (dict) at a time to avoid loading everything into memory.

    Args:
        repo_owner (str): GitHub username or organization
        repo_name (str): Repository name
        branch (str): Branch name (default: main)
    """
    url = f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/{branch}"
    resp = requests.get(url)

    if resp.status_code == 404 and branch == "main":
        # Try fallback to master
        yield from read_repo_data(repo_owner, repo_name, branch="master")
        return

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: HTTP {resp.status_code}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for file_info in zf.infolist():
            filename = file_info.filename
            if not filename.lower().endswith((".md", ".mdx")):
                continue
            try:
                with zf.open(file_info) as f_in:
                    content = f_in.read().decode("utf-8", errors="replace")
                    post = frontmatter.loads(content)
                    data = post.to_dict()
                    data.update({
                        "filename": filename,
                        "repo": repo_name,
                        "owner": repo_owner,
                        "branch": branch
                    })
                    yield data
            except Exception as e:
                logging.warning("Error processing %s: %s", filename, e)
                continue


def sliding_window(seq, size, step):
    """Yield overlapping chunks from a long string."""
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    n = len(seq)
    for i in range(0, n, step):
        yield {"start": i, "chunk": seq[i:i+size]}
        if i + size >= n:
            break


def ingest_repo(owner, repo, branch="main", chunk_size=2000, step=1000):
    """
    Download repository and create text chunks.
    
    Args:
        owner: GitHub username/organization
        repo: Repository name
        branch: Branch name (default: main)
        chunk_size: Size of each chunk in characters
        step: Step size for sliding window (overlap = chunk_size - step)
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    for doc in tqdm(read_repo_data(owner, repo, branch), desc="Processing files"):
        doc_copy = doc.copy()
        content = doc_copy.pop("content", "")
        for chunk in sliding_window(content, chunk_size, step):
            chunk.update(doc_copy)
            chunks.append(chunk)
    return chunks


def build_text_index(chunks):
    """
    Build a text search index from chunks.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        minsearch.Index object
    """
    index = Index(
        text_fields=["chunk", "title", "description", "filename"],
        keyword_fields=[]
    )
    index.fit(chunks)
    return index