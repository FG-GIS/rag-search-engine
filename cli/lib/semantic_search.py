from .search_utils import CACHE_DIR,load_movies
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Any

from torch import Tensor

def verify_model() -> None:
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")

def embed_text(text:str) -> None:
    sem_s = SemanticSearch()
    embedding = sem_s.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    sem = SemanticSearch()
    documents = load_movies()
    embeddings = sem.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query:str) -> None:
    sem = SemanticSearch()
    embedding = sem.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


class SemanticSearch:
    def __init__(self) ->None:
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int,Any] = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text:str):
        if text == "" or text.isspace():
            raise ValueError("Cannot generate embedding of empty text.")

        emb = self.model.encode([text])
        return emb[0]

    def build_embeddings(self, documents:list[dict[str,Any]]) -> Tensor:
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies,show_progress_bar=True)
        with open(self.embeddings_path,"wb") as f:
            np.save(f,self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents:list[dict[str,Any]]) -> Tensor:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)

            if len(documents) == len(self.embeddings):
                return self.embeddings

        return self.build_embeddings(documents)

