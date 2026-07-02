import argparse,json
from lib.hybrid_search import HybridSearch
from lib.search_utils import GOLDEN_PATH,load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluetion CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)"
    )
    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_PATH,"r") as f:
        g_set = json.load(f)
    
    h_search = HybridSearch(load_movies())
    k = 60
    print(f"k={k}\n\n")
    for g in g_set["test_cases"]:
        results = h_search.rrf_search(g["query"] ,k,limit)
        correct_results = 0
        doc_titles = []
        for r in results:
            doc = r["document"]
            doc_titles.append(doc["title"])
            if doc["title"] in g["relevant_docs"]:
                correct_results += 1
        patk = correct_results / args.limit
        ratk = correct_results / len(g["relevant_docs"])
        retrieved = ""
        for x in doc_titles:
            retrieved += x+", "
        relevant = ""
        for x in g["relevant_docs"]:
            relevant += x+", "
        print(f"""- Query: {g["query"]}
                    - Precision@{args.limit}: {patk:.4f}
                    - Recall@{args.limit}: {ratk:.4f}
                    - Retrieved: {retrieved}
                    - Relevant: {relevant}
        """)


if __name__ == "__main__":
    main()
