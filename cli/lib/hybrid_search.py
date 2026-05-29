import os

from cli.lib.search_utils import SearchResult

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
            combined_data[res["id"]] = {"keyword": normalized_bm25[i],"semantic": 0.0,"document":self.semantic_search.document_map[res["id"]]}

        for i,res in enumerate(semantic_results):
            if res["id"] not in combined_data:
                combined_data[res["id"]] = {"keyword": 0.0,"semantic": 0.0,"document": self.semantic_search.document_map[res["id"]]}
            combined_data[res["id"]]["semantic"] = normalized_semantic[i]

        for id, item in combined_data.items():
            hybrid = hybrid_score(item["keyword"],item["semantic"],alpha)
            combined_data[id]["hybrid"] = hybrid
        return sorted(combined_data.values(),key=lambda d: d["hybrid"],reverse=True)


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
