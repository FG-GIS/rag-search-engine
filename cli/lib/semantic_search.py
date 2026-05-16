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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self) ->None:
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents:list[dict[str,Any]] = []
        self.document_map: dict[Any,Any] = {}
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

    def search(self, query:str, limit:int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        scores = []
        for i,e in enumerate(self.embeddings):
            sim_score = cosine_similarity(query_embedding,e)
            scores.append((sim_score,self.documents[i]))

        scores = sorted(scores,key=lambda item: item[0], reverse=True)

        out = []
        for item in scores[:limit]:
            out.append({"score":item[0],"title":item[1]["title"],"description":item[1]["description"]})

        return out
