#!/usr/bin/env python
import argparse
from ingest import ingest_repo, build_text_index
from search_tools import create_embeddings, VectorSearch
from search_agent import answer_question_manual

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_argument("--query", type=str, required=True, help="Your question/query")
    parser.add_argument(
        "--search-method", 
        type=str, 
        default="hybrid",
        choices=["text", "vector", "hybrid"],
        help="Search method to use (default: hybrid)"
    )
    args = parser.parse_args()

    print("📥 Loading Evidently documentation...")
    
    # Ingest Evidently docs
    chunks = ingest_repo("evidentlyai", "docs")
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Build text index
    print("🔍 Building text search index...")
    index = build_text_index(chunks)
    
    # Build vector search
    print("🧮 Creating embeddings (this may take a few minutes)...")
    embeddings = create_embeddings(chunks)
    vindex = VectorSearch()
    vindex.fit(embeddings, chunks)
    
    print(f"\n❓ Question: {args.query}")
    print(f"🔎 Using {args.search_method} search method\n")

    # Answer the question
    answer = answer_question_manual(args.query, args.search_method, index, vindex)
    print(f"💡 Answer:\n{answer}")

if __name__ == "__main__":
    main()