import string
import math
import pickle
import os
from typing import Any, Counter
from .search_utils import BM25_B, DEFAULT_SEARCH_LIMIT, CACHE_DIR, load_movies, load_stop_words, BM25_K1
from nltk.stem import PorterStemmer

def search_command(query:str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()
    prep_query = process_text(query)
    results = []
    seen = set()
    for q in prep_query:
        found_ids = index.get_documents(q)
        for id in found_ids:
            if id in seen:
                continue
            seen.add(id)
            doc = index.docmap[id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results

def process_text(text:str) -> list[str]:
    return stem_words(remove_stop_words(tokenize_text(preprocess_text(text))))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def tokenize_text(text:str) -> list[str]:
    output = text.split()
    output = [item for item in output if item.strip()]
    return output

def remove_stop_words(t_list:list[str]) -> list[str]:
    stop_words = load_stop_words()
    filtered = [token for token in t_list if token not in stop_words]
    return filtered

def stem_words(list: list[str]) -> list[str]:
    result = []
    stemmer = PorterStemmer()
    for item in list:
        result.append(stemmer.stem(item))
    return result

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id:int, term:str, k1:float = BM25_K1,b:float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

class InvertedIndex:

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int,Any] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequency_path = os.path.join(CACHE_DIR, "term_frequency.pkl")
        self.term_frequency: dict[int, Counter] = {}
        self.doc_lengths: dict[int,int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = process_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for t in tokens:
            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()
            self.term_frequency[doc_id][t] += 1
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0
        return sum(self.doc_lengths.values())/len(self.doc_lengths)

    def get_documents(self, term:str) -> list[int]:
        term = process_text(term)[0]
        if term not in self.index:
            return []
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        token = process_text(term)
        if len(token) > 1:
            raise ValueError("Term must be a single token.")
        return self.term_frequency[doc_id][token[0]]

    def get_idf(self, term: str) -> float:
        token = process_text(term)
        if len(token) > 1:
            raise ValueError("Term must be a single token.")
        return math.log((len(self.docmap) + 1) / (len(self.get_documents(token[0])) + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = process_text(term)
        if len(token) > 1:
                raise ValueError("Term must be a single token.")
        df = len(self.get_documents(token[0]))
        N = len(self.docmap)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id:int, term:str, k1:float = BM25_K1, b:float = BM25_B) -> float:
        token = process_text(term)
        if len(token) > 1:
            raise ValueError("Term must be a single token.")
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, token[0])
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id:int, term:str) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id,term)

    def bm25_search(self, query:str, limit:int) -> list[dict[int,float]]:
        q_tkns = process_text(query)
        scores: dict[int,float] = {}
        for id in self.docmap:
            total = 0
            for q in q_tkns:
                total += self.bm25(id, q)
            scores[id] = total
        return [dict(sorted(scores.items(), key=lambda item: item[1],reverse=True)[:limit])]



    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open(self.index_path,"wb") as f:
            pickle.dump(self.index,f)
        with open(self.docmap_path,"wb") as f:
            pickle.dump(self.docmap,f)
        with open(self.term_frequency_path,"wb") as f:
            pickle.dump(self.term_frequency,f)
        with open(self.doc_lengths_path,"wb") as f:
            pickle.dump(self.doc_lengths,f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequency_path, "rb") as f:
            self.term_frequency = pickle.load(f)
        with open(self.doc_lengths_path,"rb") as f:
            self.doc_lengths = pickle.load(f)
