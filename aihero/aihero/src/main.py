#!/usr/bin/env python
import argparse
from aihero.aihero.src.ingest import ingest_repo, build_text_index
from aihero.aihero.src.search_tools import create_embeddings, VectorSearch, text_search, vector_search, hybrid_search
from aihero.aihero.src.search_agent import answer_question_manual

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_argument("--query", type=str, required=True, help="Your question/query")
    args = parser.parse_args()

    # Ingest Evidently docs
    chunks = ingest_repo("evidentlyai", "docs")
    index = build_text_index(chunks)

    # Build vector search
    embeddings = create_embeddings(chunks)
    vindex = VectorSearch()
    vindex.fit(embeddings, chunks)

    # Answer the question
    answer = answer_question_manual(args.query, "hybrid", index, vindex)
    print(f"\n💡 Answer:\n{answer}")

if __name__ == "__main__":
    main()
