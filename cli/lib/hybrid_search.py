import os

from lib.search_utils import SearchResult

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def normalize_scores(scores: list[float]) -> list[float]:
    max_score = max(scores)
    min_score = min(scores)
    if min_score == max_score:
        return [1.0 for _ in range(len(scores))]

    out = []
    for s in scores:
        out.append((s - min_score)/(max_score - min_score))

    return out

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm_25_results = self._bm25_search(query,limit*500)
        bm_25_scores = []
        for res in bm_25_results:
            bm_25_scores.append(res["score"])
        normalized_bm25 = normalize_scores(bm_25_scores)

        semantic_results = self.semantic_search.search_chunks(query,limit*500)
        semantic_scores = []
        for res in semantic_results:
            semantic_scores.append(res["score"])
        normalized_semantic = normalize_scores(semantic_scores)

        combined_data = {}
        for i,res in enumerate(bm_25_results):
            combined_data[res["id"]] = {"keyword": normalized_bm25[i],"semantic": 0.0,"document":self.idx.docmap[res["id"]]}

        for i,res in enumerate(semantic_results):
            if res["id"] not in combined_data:
                combined_data[res["id"]] = {"keyword": 0.0,"semantic": 0.0,"document": self.idx.docmap[res["id"]]}
            combined_data[res["id"]]["semantic"] = normalized_semantic[i]

        for id, item in combined_data.items():
            hybrid = hybrid_score(item["keyword"],item["semantic"],alpha)
            combined_data[id]["hybrid"] = hybrid
        return sorted(combined_data.values(),key=lambda d: d["hybrid"],reverse=True)[:limit]


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm_25_results = self._bm25_search(query,limit*500)

        semantic_results = self.semantic_search.search_chunks(query,limit*500)

        rrf = {}
        for i,res in enumerate(bm_25_results,start=1):
            rrf[res["id"]] = {"document":self.idx.docmap[res["id"]],"bm_25_rank":i,"semantic_rank":0}

        for i,res in enumerate(semantic_results,start=1):
            if res["id"] not in rrf:
                rrf[res["id"]] = {"document":self.idx.docmap[res["id"]],"bm_25_rank":0}
            rrf[res["id"]]["semantic_rank"] = i

        for item in rrf.values():
            bm25_rrf = 0
            semantic_rrf = 0

            if item["bm_25_rank"] != 0:
                bm25_rrf = rrf_score(item["bm_25_rank"],k)

            if item["semantic_rank"] != 0:
                semantic_rrf = rrf_score(item["semantic_rank"],k) 

            item["rrf"] = bm25_rrf + semantic_rrf


        return sorted(rrf.values(), key=lambda item: item["rrf"], reverse=True)[:limit]

