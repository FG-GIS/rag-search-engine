from typing import Any
from .keyword_search import process_text
from .search_utils import load_movies
from pickle import dump
import os

class InvertedIndex:

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int,Any] = {}

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = process_text(text)
        for t in tokens:
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)

    def get_documents(self, term:str) -> list[int]:
        term = term.lower()
        if term not in self.index:
            return []
        return sorted(self.index[term])

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open('cache/index.pkl',"wb") as f:
            dump(self.index,f)
        with open('cache/docmap.pkl',"wb") as f:
            dump(self.docmap,f)
