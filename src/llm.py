"""LLM answer generation using Groq."""

import os
from typing import List

from langchain_groq import ChatGroq


class GroqLLM:
    """Thin wrapper around ChatGroq for context-grounded answering."""

    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Put it in a .env file or export it."
            )
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024,
        )
        print(f"Initialized Groq LLM: {model_name}")

    def answer(self, query: str, contexts: List[str]) -> str:
        context = "\n\n".join(contexts)
        prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer the "
            "question. If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return self.llm.invoke(prompt).content
