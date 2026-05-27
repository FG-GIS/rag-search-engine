import argparse
from lib.hybrid_search import normalize_scores

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the list of given floating point scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of scores to normalize")

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
