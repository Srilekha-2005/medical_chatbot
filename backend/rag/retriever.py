"""Retriever that queries the vector store for relevant medical content."""

from typing import Optional

from backend.config import RETRIEVAL_TOP_K
from backend.rag.vector_store import VectorStore


class Retriever:
    """Retrieves relevant medical knowledge for a user query."""

    def __init__(self, vector_store: Optional[VectorStore] = None, top_k: int = RETRIEVAL_TOP_K):
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve top-k relevant chunks. Returns list of dicts with keys:
        text, metadata, score.
        """
        k = top_k if top_k is not None else self.top_k
        results = self.vector_store.search(query, top_k=k)
        return [
            {"text": text, "metadata": meta, "score": score}
            for text, meta, score in results
        ]

    def set_vector_store(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
