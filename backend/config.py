"""Configuration for the AI Health Education Assistant backend."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "processed"

# Dataset paths
MEDICAL_KNOWLEDGE_PATH = DATASETS_DIR / "medical_knowledge.json"
PHYSIO_SIGNALS_PATH = DATASETS_DIR / "physio_signals.csv"
SLEEP_DATA_PATH = DATASETS_DIR / "sleep_data.csv"
VITAL_SIGNS_PATH = DATASETS_DIR / "vital_signs.csv"
SIMPLIFICATION_PAIRS_PATH = DATASETS_DIR / "simplification_pairs.csv"

# RAG settings
VECTOR_STORE_COLLECTION = "medical_knowledge"
RETRIEVAL_TOP_K = 5
RAG_TOP_K = 3  # Number of similar Q&A entries to return as context in RAG pipeline
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight sentence transformer

# API settings
API_TITLE = "AI Health Education Assistant API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Multi-agent backend for health education: retrieval and insight generation."

# Health Insight Agent: LLM for patient-friendly explanations (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
INSIGHT_LLM_MODEL = "llama3.2:8b"  # LLaMA-3-8B-Instruct via Ollama
MEDICAL_DISCLAIMER = (
    "This information is for educational purposes only and is not medical advice, "
    "diagnosis, or treatment. Always consult a healthcare provider for personal health decisions."
)


def ensure_datasets_dir() -> Path:
    """Ensure datasets directory exists."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR
