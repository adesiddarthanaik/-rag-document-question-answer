"""Query-time retrieval from the vector store."""

from typing import Any, Dict, List

from .embeddings import EmbeddingManager
from .vectorstore import VectorStore


class Retriever:
    """Embeds a query and fetches the most relevant chunks."""

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed([query])[0]
        results = self.vector_store.query(query_emb, top_k=top_k)
        return [r for r in results if r["similarity_score"] >= min_score]
