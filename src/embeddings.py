"""Embedding generation using SentenceTransformer."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Generates dense embeddings for text using a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dim}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) array of embeddings for `texts`."""
        return self.model.encode(texts, show_progress_bar=len(texts) > 1)
