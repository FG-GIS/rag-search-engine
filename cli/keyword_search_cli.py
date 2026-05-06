import argparse

from lib.keyword_search import search_command,InvertedIndex,bm25_idf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds the movie index database")

    tf_parser = subparsers.add_parser("tf", help="Token frequency command, usage tf <doc_id> <term>")
    tf_parser.add_argument("doc_id", type=int, help="Term frequency document")
    tf_parser.add_argument("term", type=str, help="Term to check")

    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency command, usage idf <term>")
    idf_parser.add_argument("term", type=str, help="Term to check")

    tfidf_parser = subparsers.add_parser("tfidf", help="Inverse document frequency command, usage idf <term>")
    tfidf_parser.add_argument("doc_id", type=int, help="Term frequency document")
    tfidf_parser.add_argument("term", type=str, help="Term to check")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

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
            idf = index.get_idf(args.term)
            print(f"Docmap length: {len(index.docmap)}\nTerm per doc count: {len(index.get_documents(args.term))}")
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tf_idf = index.get_tf(args.doc_id,args.term) * index.get_idf(args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
