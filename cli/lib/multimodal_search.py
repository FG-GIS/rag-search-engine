from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies
from .semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self, documents: list = [], model_name = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name_or_path=model_name)
        self.documents = documents
        self.texts: list[str] = []

        for d in documents:
            self.texts.append(f"{d['title']}: {d['description']}")

        self.text_embeddings = self.model.encode(self.texts,show_progress_bar=True)

    def embed_image(self, img_path: str):
        img = Image.open(img_path)
        return self.model.encode(img)

    def search_from_image(self, img_path: str) -> list[dict]:
        img = Image.open(img_path)
        img_emb = self.model.encode(img)
        cos_sim_list = []
        out = []

        for e_t in self.text_embeddings:
            cos_sim_list.append(cosine_similarity(e_t,img_emb))

        for i in range(len(cos_sim_list)):
            d = self.documents[i]
            out.append({
                "id": d['id'],
                "title": d['title'],
                "description": d['description'],
                "score": cos_sim_list[i],
            })
        return sorted(out,key=lambda item: item['score'], reverse=True)[:5]

def verify_image_embedding(img_path: str):
    mmsearch = MultimodalSearch()
    embedding = mmsearch.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def search_by_image(img_path: str) -> list[dict]:
    mms = MultimodalSearch(documents=load_movies())
    return mms.search_from_image(img_path)
