from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def embed_image(self, img_path: str):
        img = Image.open(img_path)
        return self.model.encode(img)

def verify_image_embedding(img_path: str):
    mmsearch = MultimodalSearch()
    embedding = mmsearch.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
