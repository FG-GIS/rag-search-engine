import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies,simple_query
def print_results(results: list) -> list[str]:
    formatted_results = []
    print("Search Results:")
    for i,r in enumerate(results, start=1):
        line = f"{i+1}. {r["document"]["title"]}\nRRF Score: {r["rrf"]}\nBM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}\n{r["document"]["description"][:100]}"
        formatted_results.append(line)
        print(f"- {r["document"]["title"]}")
    return formatted_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser("summarize", help="Perform RAG and summarize response.")
    summary_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument("--limit", type=int, default=5, help="Max number of results.")

    args = parser.parse_args()

    query = args.query
    hybrid = HybridSearch(load_movies())

    match args.command:
        case "rag":
            results = hybrid.rrf_search(query=query,limit=5)

            formatted_results = print_results(results)

            rag_response = simple_query(query,formatted_results,"rag")

            print("\nRAG Response:")
            print(rag_response)

        case "summarize":
            results = hybrid.rrf_search(query=query, limit=args.limit)

            formatted_results = print_results(results)

            summary_response = simple_query(query,formatted_results,"summary")

            print("\nLLM Summary:")
            print(summary_response)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
