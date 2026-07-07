import argparse
from lib.search_utils import evaluate_results, load_movies,enhance_query_spelling,rewrite_query,expand_query,rerank_results,batch_rerank_results,cross_encoding.evaluate_results
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
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual","batch","cross_encoder"], help="Search enhancement method")
    rrf_search_parser.add_argument("--evaluate", type=bool, help="Rate the search results")
    rrf_search_parser.add_argument("--debug", type=bool, help="Enable debug prints")

    args = parser.parse_args()

    limit = args.limit
    debug = args.debug

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
            if debug:
                print(f"|DEBUG --> query: {query}")
            match args.enhance:
                case "spell":
                    query = enhance_query_spelling(query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "rewrite":
                    query = rewrite_query(query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "expand":
                    query = expand_query(query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            if debug and args.enhance is not None:
                print(f"|DEBUG --> enhanced query: {query}")

            match args.rerank_method:
                case "individual" | "batch" | "cross_encoder":
                    limit = limit * 5

            results = h_search.rrf_search(args.query + " " + query,args.k,limit)
            if debug:
                for i,r in enumerate(results):
                    print(f"""|DEBUG --> results: {i+1}. {r["document"]["title"]}
|DEBUG --> results: RRF Score: {r["rrf"]}
|DEBUG --> results: BM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}
|DEBUG --> results: {r["document"]["description"][:100]}\n""")

            formatted_results = []
            match args.rerank_method:
                case "individual":
                    results = rerank_results(query, results)[:args.limit]
                    print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                    print(f"Reciprocal Rank Fusion for '{args.query}' (k={args.k})")
                    for i,r in enumerate(results):
                        formatted_results.append(f"{i+1}. {r["document"]["title"]}\nRe-rank Score: {r["re_ranking"]}/10\nRRF Score: {r["rrf"]}\nBM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}\n{r["document"]["description"][:100]}")

                case "batch":
                    results = batch_rerank_results(query, results)[:args.limit]
                    print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                    print(f"Reciprocal Rank Fusion for '{args.query}' (k={args.k})")
                    for i,r in enumerate(results):
                        formatted_results.append(f"{i+1}. {r["document"]["title"]}\nRe-rank Rank: {r["batch_re_ranking"]}\nRRF Score: {r["rrf"]}\nBM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}\n{r["document"]["description"][:100]}")

                case "cross_encoder":
                    pairs = []
                    for r in results:
                        doc = r["document"]
                        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
                    cross_scores = cross_encoding(pairs).tolist()
                    cross_results = []
                    for i,r in enumerate( results ):
                        r["cross_score"] = cross_scores[i] 
                        cross_results.append(r)
                    cross_results = sorted(cross_results, key= lambda item: item["cross_score"],reverse=True)[:args.limit]
                    for i,r in enumerate(cross_results):
                        formatted_results.append(f"{i+1}. {r["document"]["title"]}\nCross Encoder Score: {r["cross_score"]}\nRRF Score: {r["rrf"]}\nBM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}\n{r["document"]["description"][:100]}")


                case _:
                    for i,r in enumerate(results):
                        formatted_results.append(f"{i+1}. {r["document"]["title"]}\nRRF Score: {r["rrf"]}\nBM25 Rank: {r["bm_25_rank"]}, Semantic Rank: {r["semantic_rank"]}\n{r["document"]["description"][:100]}")
            for l in formatted_results:
                print(l)

            ev_list = []
            if args.evaluate:
                scores = evaluate_results(query, formatted_results)
                for i in range(len(results)):
                    ev_list.append("Title":{results[i]["document"]["title"],"Score":scores[i]})

                ev_list = sorted(ev_list,lambda item: item[1],reverse=True)
                for i,r in enumerate(ev_list):
                    print(f"{i}. {r["Title"]}: {r["Score"]}/3")



        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
