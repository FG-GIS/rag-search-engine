import string
import pickle
import os
from typing import Any, Counter
from .search_utils import DEFAULT_SEARCH_LIMIT, CACHE_DIR, load_movies, load_stop_words
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
    # for w in stop_words:
    #     if w in t_list:
    #         t_list.remove(w)
    return filtered

def stem_words(list: list[str]) -> list[str]:
    result = []
    stemmer = PorterStemmer()
    for item in list:
        result.append(stemmer.stem(item))
    return result


class InvertedIndex:

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int,Any] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequency_path = os.path.join(CACHE_DIR, "term_frequency.pkl")
        self.term_frequency: dict[int, Counter] = {}

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = process_text(text)
        for t in set(tokens):
            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()
            self.term_frequency[doc_id][t] += 1
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)

    def get_documents(self, term:str) -> list[int]:
        term = process_text(term)[0]
        if term not in self.index:
            return []
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        token = process_text(term)
        if len(token) > 1:
            raise Exception("Too many tokens.")
        return self.term_frequency[doc_id][token[0]]

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

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequency_path, "rb") as f:
            self.term_frequency = pickle.load(f)
