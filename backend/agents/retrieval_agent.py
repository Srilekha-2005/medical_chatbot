"""Data Retrieval Agent: loads datasets and retrieves medical context using FAISS + sentence-transformers."""

import re
from dataclasses import asdict
from typing import Any, Optional

from backend.data_loader import (
    load_medical_knowledge,
    load_sleep_data,
    load_vital_signs,
    load_physiological_signals,
)
from backend.rag.faiss_store import FAISSVectorStore
from backend.config import RETRIEVAL_TOP_K


def _looks_like_rare_disease_entry(question: str, text: str) -> bool:
    """
    Heuristic: True if this entry appears to be about a rare/single disease
    (e.g. Liddle syndrome) rather than general health education.
    Avoid returning these for cause_question so we prefer general explanations.
    """
    q = (question or "").strip().lower()
    t = (text or "").lower()[:500]
    # Specific disease/syndrome name in question (e.g. "What is Liddle syndrome?")
    if re.search(r"\bwhat\s+is\s+\w+\s+syndrome\b", q):
        return True
    if re.search(r"\bwhat\s+is\s+\w+\s+disease\b", q):
        return True
    if re.search(r"\w+\s+syndrome\s*\??\s*$", q):
        return True
    # Question is mainly a proper-name condition (single rare entity)
    if re.search(r"^(what is|define)\s+[A-Za-z]+\s+syndrome", q, re.IGNORECASE):
        return True
    return False


def _expand_query_for_cause_question(query: str) -> str:
    """Bias retrieval toward general causes, lifestyle, risk factors instead of rare diseases."""
    return (
        "general causes lifestyle risk factors common causes "
        + (query or "").strip()
    )


def _extract_disease_name_from_query(query: str) -> str:
    """
    Extract probable disease name from a definition-style query, e.g.
    "What is Marfan syndrome?" -> "marfan syndrome".
    """
    if not query:
        return ""
    q = query.strip().lower()
    for prefix in ("what is ", "what are ", "define ", "explain "):
        if q.startswith(prefix):
            q = q[len(prefix) :].strip()
            break
    q = q.rstrip("?.").strip()
    return q


class DataRetrievalAgent:
    """
    Agent responsible for:
    1. Loading datasets via the dataset loader (medical_knowledge, physio_signals, sleep_data, vital_signs)
    2. Retrieving medical knowledge from medical_knowledge.json
    3. Retrieving physiological signals from physio_signals.csv
    4. Retrieving sleep metrics from sleep_data.csv
    5. Retrieving vital signs from vital_signs.csv

    Uses sentence-transformers embeddings and FAISS vector search for medical context retrieval.
    """

    def __init__(self, top_k: int = RETRIEVAL_TOP_K):
        self.top_k = top_k
        self._faiss_store: Optional[FAISSVectorStore] = None
        self._index_loaded = False

    def _ensure_medical_index(self) -> None:
        """Load medical_knowledge from dataset loader and build FAISS index with sentence-transformers."""
        if self._index_loaded:
            return
        knowledge = load_medical_knowledge()
        if not knowledge:
            self._faiss_store = FAISSVectorStore()
            self._index_loaded = True
            return
        self._faiss_store = FAISSVectorStore()
        texts = []
        metadatas = []
        for entry in knowledge:
            text = f"Q: {entry.question}\nA: {entry.answer}"
            texts.append(text)
            metadatas.append({"question": entry.question, "source": "medical_knowledge"})
        self._faiss_store.add_documents(texts, metadatas)
        self._index_loaded = True

    def retrieve_medical_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        question_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant medical answers for the query using sentence-transformers
        embeddings and FAISS vector search.

        When question_type is cause_question, prefers general health education (lifestyle,
        risk factors) and avoids rare/single-disease entries (e.g. Liddle syndrome).

        Returns list of dicts with keys: text, metadata, score (cosine similarity).
        """
        k = top_k if top_k is not None else self.top_k
        self._ensure_medical_index()
        if self._faiss_store is None or self._faiss_store.num_documents == 0:
            return []

        if question_type == "cause_question":
            search_query = _expand_query_for_cause_question(query)
            # Retrieve more candidates so we can filter out rare-disease entries
            fetch_k = min(k * 4, 20)
            results = self._faiss_store.search(search_query, top_k=fetch_k)
            out = []
            for text, meta, score in results:
                question = (meta or {}).get("question", "")
                if _looks_like_rare_disease_entry(question, text):
                    continue
                out.append({"text": text, "metadata": meta, "score": score})
                if len(out) >= k:
                    break
            if not out:
                results = self._faiss_store.search(search_query, top_k=k)
                out = [{"text": text, "metadata": meta, "score": score} for text, meta, score in results]
            return out

        if question_type == "disease_specific_question":
            # For disease definitions, bias the query toward definition/symptoms and
            # prefer entries whose question closely matches the disease name.
            disease_name = _extract_disease_name_from_query(query)
            search_query = f"{query} disease definition symptoms".strip()
            fetch_k = min(k * 4, 20)
            results = self._faiss_store.search(search_query, top_k=fetch_k)
            if not results:
                results = self._faiss_store.search(query, top_k=k)
            scored: list[dict[str, Any]] = []
            disease_lower = disease_name or query.strip().lower()
            for text, meta, score in results:
                q_text = ((meta or {}).get("question") or "").lower()
                # Simple similarity: prefer when disease name is a substring of the stored question.
                name_match = 1.0 if disease_lower and disease_lower in q_text else 0.0
                scored.append(
                    {
                        "text": text,
                        "metadata": meta,
                        "score": float(score) + name_match,  # boost exact-name matches
                    }
                )
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:k]

        results = self._faiss_store.search(query, top_k=k)
        return [
            {"text": text, "metadata": meta, "score": score}
            for text, meta, score in results
        ]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        question_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Retrieve relevant medical context (FAISS + sentence-transformers) and optionally
        attach physiological signals, sleep metrics, and vital signs from the dataset loader.

        question_type: when "cause_question", retrieval prefers general causes and avoids
        rare disease entries (e.g. Liddle syndrome).
        """
        rag_results = self.retrieve_medical_context(
            query, top_k=top_k, question_type=question_type
        )
        out: dict[str, Any] = {"rag_results": rag_results}

        # Load and attach structured data from dataset loader
        q_lower = query.lower()
        if any(k in q_lower for k in ("sleep", "insomnia", "apnea", "oxygen")):
            sleep_records = load_sleep_data()[:50]
            out["sleep_data"] = [asdict(r) for r in sleep_records]
        else:
            out["sleep_data"] = None
        if any(k in q_lower for k in ("vital", "heart rate", "blood pressure", "temp", "spo2")):
            vital_records = load_vital_signs()[:100]
            out["vital_signs"] = [asdict(r) for r in vital_records]
        else:
            out["vital_signs"] = None
        if any(k in q_lower for k in ("physio", "bmi", "glucose", "hemoglobin", "hypertension", "diabetes")):
            physio_records = load_physiological_signals()[:50]
            out["physio_signals"] = [asdict(r) for r in physio_records]
        else:
            out["physio_signals"] = None
        return out
