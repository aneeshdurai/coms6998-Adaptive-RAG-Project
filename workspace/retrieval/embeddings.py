from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from workspace.utils.logging_utils import get_logger

logger = get_logger("retrieval.embeddings")

class Embedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embs.astype("float32")
