"""FAISS vector store with sentence-transformers for medical knowledge retrieval."""

from typing import Optional

import numpy as np

from backend.config import EMBEDDING_MODEL


class FAISSVectorStore:
    """
    Vector store using sentence-transformers for embeddings and FAISS for similarity search.
    Embeddings are L2-normalized so IndexFlatIP returns cosine similarity.
    """

    def __init__(self):
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._index = None  # faiss.IndexFlatIP
        self._model = None
        self._dim: Optional[int] = None

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2-normalize for cosine similarity via inner product."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x.astype(np.float32) / norms

    def add_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> None:
        """Encode texts with sentence-transformers, L2-normalize, and add to FAISS index."""
        if not texts:
            return
        model = self._get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self._dim = embeddings.shape[1]
        embeddings = self._normalize(embeddings)
        self._documents.extend(texts)
        self._metadata.extend(metadatas or [{}] * len(texts))
        import faiss
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
        """
        Encode query with sentence-transformers, search FAISS. Returns list of (text, metadata, score).
        Score is cosine similarity (higher = more similar).
        """
        if not self._documents or self._index is None:
            return []
        model = self._get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True)
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        q_emb = self._normalize(q_emb)
        k = min(top_k, len(self._documents))
        score_matrix, index_matrix = self._index.search(q_emb, k)
        scores = score_matrix[0]
        indices = index_matrix[0]
        return [
            (self._documents[i], self._metadata[i], float(scores[j]))
            for j, i in enumerate(indices)
            if 0 <= i < len(self._documents)
        ]

    @property
    def num_documents(self) -> int:
        return len(self._documents)
