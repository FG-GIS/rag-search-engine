import argparse
import math

from lib.keyword_search import search_command,InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Builds the movie index database")

    search_parser = subparsers.add_parser("tf", help="Token frequency command, usage tf <doc_id> <term>")
    search_parser.add_argument("doc_id", type=int, help="Term frequency document")
    search_parser.add_argument("term", type=str, help="Term to check")

    search_parser = subparsers.add_parser("idf", help="Inverse document frequency command, usage idf <term>")
    search_parser.add_argument("term", type=str, help="Term to check")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results,1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            index = InvertedIndex()
            index.build()
            index.save()
            print("Inverted index built successfully!")
        case "tf":
            index = InvertedIndex()
            index.load()
            print(f"Document: {args.doc_id}\nTerm: {args.term}\nCount: {index.get_tf(args.doc_id,args.term)}")
        case "idf":
            index = InvertedIndex()
            index.load()
            idf = math.log((len(index.docmap) + 1) / (len(index.get_documents(args.term)) + 1))
            print(f"Docmap length: {len(index.docmap)}\nTerm per doc count: {len(index.get_documents(args.term))}")
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
