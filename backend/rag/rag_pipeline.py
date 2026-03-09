"""
Retrieval Augmented Generation pipeline.

1. Load medical knowledge question-answer pairs.
2. Convert questions to embeddings using sentence-transformers.
3. Store embeddings in FAISS vector index.
4. On query: embed the query, retrieve top-k similar entries (by question).
5. Return retrieved answers as context.
"""

from typing import Optional

import numpy as np

from backend.config import EMBEDDING_MODEL, RAG_TOP_K
from backend.data_loader import load_medical_knowledge


class RAGPipeline:
    """
    RAG pipeline: index medical Q&A by question embeddings (FAISS),
    then return retrieved answers as context for a user query.
    """

    def __init__(self, top_k: int = RAG_TOP_K):
        self.top_k = top_k
        self._questions: list[str] = []
        self._answers: list[str] = []
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
        """L2-normalize for cosine similarity via FAISS IndexFlatIP."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x.astype(np.float32) / norms

    def build_index(self) -> None:
        """
        Step 1–3: Load medical Q&A pairs, embed questions with sentence-transformers,
        store embeddings in FAISS vector index. Keeps answers in order for retrieval.
        """
        pairs = load_medical_knowledge()
        if not pairs:
            self._questions = []
            self._answers = []
            return
        questions = [p.question for p in pairs]
        answers = [p.answer for p in pairs]
        model = self._get_embedding_model()
        embeddings = model.encode(questions, convert_to_numpy=True)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self._dim = embeddings.shape[1]
        embeddings = self._normalize(embeddings)
        self._questions = questions
        self._answers = answers
        import faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)

    def _ensure_index(self) -> None:
        """Build index once on first use."""
        if self._index is None and not self._questions:
            self.build_index()

    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Step 4–5: Embed the query, retrieve top-k similar entries (by question),
        return retrieved answers as a single context string.
        """
        k = top_k if top_k is not None else self.top_k
        self._ensure_index()
        if not self._answers or self._index is None:
            return ""
        model = self._get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True)
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        q_emb = self._normalize(q_emb)
        k = min(k, len(self._answers))
        score_matrix, index_matrix = self._index.search(q_emb, k)
        indices = index_matrix[0]
        retrieved_answers = [
            self._answers[i] for i in indices
            if 0 <= i < len(self._answers)
        ]
        return "\n\n".join(retrieved_answers)

    def get_context_entries(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Same as get_context but returns a list of dicts with question, answer, and score
        for each retrieved entry (useful when structure is needed).
        """
        k = top_k if top_k is not None else self.top_k
        self._ensure_index()
        if not self._answers or self._index is None:
            return []
        model = self._get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True)
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        q_emb = self._normalize(q_emb)
        k = min(k, len(self._answers))
        score_matrix, index_matrix = self._index.search(q_emb, k)
        scores = score_matrix[0]
        indices = index_matrix[0]
        return [
            {
                "question": self._questions[i],
                "answer": self._answers[i],
                "score": float(scores[j]),
            }
            for j, i in enumerate(indices)
            if 0 <= i < len(self._answers)
        ]
