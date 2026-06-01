"""End-to-end RAG pipeline orchestrator."""

from typing import Any, Dict

from .embeddings import EmbeddingManager
from .ingestion import load_pdfs, split_documents
from .llm import GroqLLM
from .retriever import Retriever
from .vectorstore import VectorStore


class RAGPipeline:
    """Ties ingestion, embedding, storage, retrieval and generation together."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "data/vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemma2-9b-it",
    ) -> None:
        self.embedder = EmbeddingManager(embedding_model)
        self.store = VectorStore(collection_name, persist_directory)
        self.retriever = Retriever(self.store, self.embedder)
        self.llm = GroqLLM(llm_model)

    def ingest(self, pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Load PDFs, chunk them, embed, and store. Returns chunk count."""
        docs = load_pdfs(pdf_dir)
        if not docs:
            print("No documents found to ingest.")
            return 0
        chunks = split_documents(docs, chunk_size, chunk_overlap)
        embeddings = self.embedder.embed([c.page_content for c in chunks])
        self.store.add(chunks, embeddings)
        return len(chunks)

    def ask(self, question: str, top_k: int = 4, min_score: float = 0.0) -> Dict[str, Any]:
        """Retrieve context and generate a grounded answer with sources."""
        results = self.retriever.retrieve(question, top_k=top_k, min_score=min_score)
        if not results:
            return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0}

        answer = self.llm.answer(question, [r["content"] for r in results])
        sources = [
            {
                "source": r["metadata"].get("source_file", "unknown"),
                "page": r["metadata"].get("page", "?"),
                "score": round(r["similarity_score"], 3),
            }
            for r in results
        ]
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(max(r["similarity_score"] for r in results), 3),
        }
