from .search_utils import CACHE_DIR, format_search_result,load_movies
import re,json,os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Any


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

def chunker(text:str,size:int,overlap:int,semantic_flag:bool = False) -> list[str]:
    text = text.strip()
    if text == "":
        return []
    if size == 0 or size - overlap <= 0:
        raise ValueError("Size or overlap value error.")
    if semantic_flag:
        tokens = re.split(r"(?<=[.!?])\s+",text)
        if len(tokens) == 1:
            if not tokens[0].endswith(('.','?','!')):
                tokens = [text]
    else:
        tokens = text.split()
    tk2 = []
    for i in range(len(tokens)):
        stripped_chunk = tokens[i].strip()
        if stripped_chunk != "":
            tk2.append(tokens[i])
    tokens = [tokens[i].strip() for i in range (len(tokens))]
    sub_chunks = [tokens[i:i+size] for i in range(0, len(tokens),size - overlap)]
    if len(sub_chunks[len(sub_chunks)-1]) == overlap:
        sub_chunks = sub_chunks[:-1]
    out = [' '.join(sub_chunks[i]) for i in range(len(sub_chunks))]
    return out

def chunk_printer(text:str,chunks:list[str],sem_flag:bool = False):
    if sem_flag:
        print(f"Semantically chunking {len(text)} characters")
    else:
        print(f"Chunking {len(text)} characters")
    for i,c in enumerate(chunks):
        print(f"{i+1}. {c}")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") ->None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents:list[dict[str,Any]] = []
        self.document_map: dict[Any,Any] = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text:str):
        if text == "" or text.isspace():
            raise ValueError("Cannot generate embedding of empty text.")

        emb = self.model.encode([text])
        return emb[0]

    def build_embeddings(self, documents:list[dict[str,Any]]):
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies,show_progress_bar=True)
        with open(self.embeddings_path,"wb") as f:
            np.save(f,self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents:list[dict[str,Any]]):
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents:list[dict[str,Any]]):
        self.documents = documents
        movies = []
        moveies_dict = []
        for i,doc in enumerate(documents):
            self.document_map[doc["id"]] = doc

            if doc['description'].isspace():
                continue
            split_desc = chunker(doc['description'],4,1,True)
            for j,desc_chunk in enumerate(split_desc):
                movies.append(desc_chunk)
                moveies_dict.append({"movie_idx":i,"chunk_idx":j,"total_chunks":len(split_desc)})

        self.chunk_metadata = moveies_dict
        self.chunk_embeddings = self.model.encode(movies,convert_to_numpy=True,show_progress_bar=True)
        with open(self.chunk_embeddings_path,"wb") as f:
            np.save(f,self.chunk_embeddings)

        with open(self.metadata_path,"w") as f:
            json.dump({"chunks": moveies_dict, "total_chunks": len(movies)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.metadata_path):
            with open(self.chunk_embeddings_path, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(self.metadata_path,"r") as f:
                self.chunk_metadata = json.load(f)['chunks']
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("Error Embeddings not loaded.`")
        if self.chunk_metadata is None:
            raise ValueError("Error Metadata not loaded.`")

        scores_dict = []
        q_emb = self.generate_embedding(query)

        for i,e in enumerate(self.chunk_embeddings):
            score = cosine_similarity(q_emb, e)
            current_meta = self.chunk_metadata[i]
            scores_dict.append({"chunk_idx":current_meta['chunk_idx'],
                                "movie_idx":current_meta['movie_idx'],
                                "score":score})

        movie_x_scores = {}
        for s in scores_dict:
            if s['movie_idx'] in movie_x_scores and movie_x_scores[s['movie_idx']] >= s['score']:
                continue
            else:
                movie_x_scores[s['movie_idx']] = s['score']

        res = sorted(movie_x_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        out = []
        for item in res:
            doc_id = item[0]
            doc = self.documents[doc_id]
            title = doc['title']
            desc = doc['description']
            score = item[1]
            metadata = self.chunk_metadata[doc_id]
            out.append(format_search_result(doc_id,title,desc,score,**metadata)) 

        return out
