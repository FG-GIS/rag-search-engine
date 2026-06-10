import argparse
from lib.search_utils import load_movies,enhance_query_spelling
from lib.hybrid_search import normalize_scores,HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the list of given floating point scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search the query using both bm25 and semantic.")
    weighted_search_parser.add_argument("query", type=str, help="Query that will be searched.")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight.")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results.")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search the query using both bm25 and semantic with rrf scoring for the results.")
    rrf_search_parser.add_argument("query", type=str, help="Query that will be searched.")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="K weight.")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results.")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            n_scores = normalize_scores(args.scores)
            for s in n_scores:
                print(f"*  {s:.4f}")
        case "weighted-search":
            h_search = HybridSearch(load_movies())
            results = h_search.weighted_search(args.query,args.alpha,args.limit)
            for i,r in enumerate(results):
                print(f"{i+1}. {r["document"]["title"]}\nHybrid Score: {r["hybrid"]}\nBM25: {r["keyword"]}, Semantic: {r["semantic"]}\n{r["document"]["description"][:100]}")
        case "rrf-search":
            h_search = HybridSearch(load_movies())
            query = args.query
            if args.enhance == "spell":
                query = enhance_query_spelling(query)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            results = h_search.rrf_search(query,args.k,args.limit)
            for i,r in enumerate(results):
                print(f"{i+1}. {r["document"]["title"]}\nRRF Score: {r["rrf"]}\nBM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}\n{r["document"]["description"][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
