# AI Health Education Assistant – Backend

FastAPI backend with a **multi-agent architecture** for health education.

## Agents

1. **Data Retrieval Agent** (`agents/retrieval_agent.py`) – Fetches relevant medical Q&A from the RAG vector store and optional structured data (sleep, vitals, physio) based on the query.
2. **Health Insight Agent** (`agents/insight_agent.py`) – Builds educational insights from retrieval results and runs the NLP pipeline on the user query.

## Project structure

```
backend/
  main.py              # FastAPI app and routes
  config.py            # Paths and settings
  agents/
    retrieval_agent.py
    insight_agent.py
  rag/
    vector_store.py    # Embeddings + similarity search
    retriever.py
  nlp_pipeline/
    morphology.py, syntax.py, semantics.py, discourse.py, pragmatics.py
  report_processing/
    report_parser.py
  data_loader/
    load_datasets.py
```

Datasets are read from `datasets/processed/` (medical_knowledge.json, physio_signals.csv, sleep_data.csv, vital_signs.csv, simplification_pairs.csv).

## Run locally

From the **project root** (`medical_chatbot/`):

```bash
# Create venv and install deps
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r backend/requirements.txt

# Optional: better retrieval (semantic search)
# pip install sentence-transformers

# Start API (PYTHONPATH so backend package resolves)
PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health:** `GET http://localhost:8000/health`
- **Query (full pipeline):** `POST http://localhost:8000/query` with body `{"query": "What is epilepsy?", "top_k": 5}`
- **Retrieve only:** `GET http://localhost:8000/retrieve?q=What is epilepsy?&top_k=5`
- **Datasets summary:** `GET http://localhost:8000/datasets`
- **OpenAPI:** `http://localhost:8000/docs`

## Configuration

Edit `backend/config.py` for dataset paths, RAG `top_k`, and embedding model (used only if `sentence-transformers` is installed).
