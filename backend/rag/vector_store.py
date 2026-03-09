"""Vector store for medical knowledge - embeddings and similarity search."""

from typing import Optional

from backend.config import EMBEDDING_MODEL, VECTOR_STORE_COLLECTION


class VectorStore:
    """In-memory vector store for medical Q&A documents with optional sentence-transformers."""

    def __init__(self, collection_name: str = VECTOR_STORE_COLLECTION):
        self.collection_name = collection_name
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._embeddings: Optional[list[list[float]]] = None
        self._model = None

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL)
            except ImportError:
                self._model = False  # Fallback: no embeddings
        return self._model if self._model else None

    def add_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> None:
        """Add documents and compute embeddings."""
        self._documents.extend(texts)
        self._metadata.extend(metadatas or [{}] * len(texts))
        model = self._get_embedding_model()
        if model is not None:
            new_embeddings = model.encode(texts).tolist()
            if self._embeddings is None:
                self._embeddings = []
            self._embeddings.extend(new_embeddings)
        else:
            self._embeddings = None

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
        """
        Search by semantic similarity. Returns list of (text, metadata, score).
        If embeddings unavailable, falls back to keyword overlap.
        """
        if not self._documents:
            return []
        model = self._get_embedding_model()
        if model is not None and self._embeddings is not None:
            query_embedding = model.encode([query])[0]
            import numpy as np
            scores = np.dot(self._embeddings, query_embedding) / (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-9
            )
            indices = np.argsort(scores)[::-1][:top_k]
            return [
                (self._documents[i], self._metadata[i], float(scores[i]))
                for i in indices
            ]
        # Fallback: simple keyword overlap
        q_lower = set(query.lower().split())
        scored = []
        for i, doc in enumerate(self._documents):
            d_lower = set(doc.lower().split())
            overlap = len(q_lower & d_lower) / (len(q_lower) + 1e-9)
            scored.append((overlap, i))
        scored.sort(key=lambda x: -x[0])
        return [
            (self._documents[i], self._metadata[i], score)
            for score, i in scored[:top_k]
        ]

    @property
    def num_documents(self) -> int:
        return len(self._documents)
