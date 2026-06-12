from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def generate_embeddings(texts):

    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    return np.array(emb)