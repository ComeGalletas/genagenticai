# Generative Agentic AI

A learning-focused agentic AI workspace built with **LangGraph**, **FastAPI**, and **React**. The agent uses a locally running **Ollama** LLM (llama3.1 / qwen3) and decides at runtime which tool to invoke to answer a question:

- **RAG (ChromaDB)** — searches a local vector knowledge base using Ollama embeddings
- **Static search** — returns deterministic results from a curated in-code index
- **Google Search** — falls back to a live Google API query when local knowledge is insufficient
- **Job Posting Search** — searches LinkedIn for recent job postings using user queries, location and remote options

The graph routes between these tools automatically, making it a good hands-on example of agentic decision-making without relying on paid cloud LLMs.

## Project structure

- `backend/`: FastAPI + LangGraph agent
- `frontend/`: React chat client

## Agentic LangGraph App

### Prerequisites

- [Ollama](https://ollama.com/) running locally with `llama3.1` (or `qwen3.6`) pulled
- Python 3.12+
- Node.js 18+

### Backend setup & run

```bash
cd backend
python -m venv .venv

.venv\Scripts\activate # Optional

pip install -e .
#copy .env.example .env
```

Start the server:

```bash
uvicorn app.main:app --reload --reload-dir app --port 8000
```

Rebuild the RAG knowledge base:

```bash
python -m app.main rebuild
```

Health check: `http://localhost:8000/health`

### Frontend setup & run

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend calls `http://localhost:8000/api/chat` by default.  
Override with `VITE_API_BASE_URL` in a `.env` file if needed.

---

