import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT,"data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT,"data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3

def load_movies() -> list[dict]:
    with open(DATA_PATH,"r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]
    #     data = f.read()
    # return data.splitlines()

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any):
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
