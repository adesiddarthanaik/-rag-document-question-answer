"""Persistent vector store backed by ChromaDB."""

import os
import uuid
from typing import Any, Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """Stores chunk embeddings + metadata in a persistent ChromaDB collection."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = "data/vector_store",
    ) -> None:
        # persist_directory=None -> in-memory store (per session, nothing on disk).
        # Use this for multi-user web deployments so users don't share documents.
        if persist_directory is None:
            self.client = chromadb.EphemeralClient()
        else:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings for RAG"},
        )
        print(
            f"Vector store ready (collection='{collection_name}', "
            f"existing docs={self.collection.count()})"
        )

    def add(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the collection."""
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        print(f"Added {len(chunks)} chunk(s). Total: {self.collection.count()}")

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the top_k most similar chunks for a query embedding."""
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )
        out: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                {
                    "content": doc,
                    "metadata": meta,
                    # Chroma returns squared-L2 distance; convert to a 0..1 score.
                    "similarity_score": 1.0 / (1.0 + dist),
                }
            )
        return out
