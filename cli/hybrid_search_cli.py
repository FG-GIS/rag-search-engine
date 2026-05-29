import argparse
from lib.hybrid_search import normalize_scores

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the list of given floating point scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search the query using both bm25 and semantic.")
    weighted_search_parser.add_argument("query", type=str, help="Query thet will be searched.")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight.")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results.")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            n_scores = normalize_scores(args.scores)
            for s in n_scores:
                print(f"*  {s:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
