#!/usr/bin/env python3
from lib.semantic_search import SemanticSearch, verify_model,embed_text,verify_embeddings,embed_query_text,load_movies,fixed_chunker

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Print embedding model data.")

    embed_parser = subparsers.add_parser("embed_text", help="Embed the given text and print out data.")
    embed_parser.add_argument("term", type=str, help="Term to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Test command to verify the embedding process")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed the given text and print out data.")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search the given query within the database")
    search_parser.add_argument("query", type=str, help="Query to search by")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search outputs")

    chunk_parser = subparsers.add_parser("chunk", help="Fixed-size chunking for the given text at given or default size")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Default chunking size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Default chunk overlap")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.term)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            sem = SemanticSearch()
            sem.load_or_create_embeddings(load_movies())
            out = sem.search(args.query,args.limit)
            for i,item in enumerate(out):
                print(f"{i}. {item['title']} ({item['score']})\n{item['description']}")
        case "chunk":
            chunks = fixed_chunker(args.text,args.chunk_size,args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i,c in enumerate(chunks):
                print(f"{i+1}. {c}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
