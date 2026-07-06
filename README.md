# Generative Agentic AI

A learning-focused agentic AI workspace built with **LangGraph**, **FastAPI**, and **React**. The agent uses a locally running **Ollama** LLM (llama3.1 / qwen3) and decides at runtime which tool to invoke to answer a question:

- **RAG (ChromaDB)** — searches a local vector knowledge base using Ollama embeddings
- **Static search** — returns deterministic results from a curated in-code index
- **Google Search** — falls back to a live Google API query when local knowledge is insufficient

The graph routes between these tools automatically, making it a good hands-on example of agentic decision-making without relying on paid cloud LLMs.

## Repository structure

```
agentic-langgraph-app/   ← main app (backend + frontend)
langgraph-course/        ← Jupyter notebooks & crash course material
```

---

## Agentic LangGraph App

### Prerequisites

- [Ollama](https://ollama.com/) running locally with `llama3.1` (or `qwen3`) pulled
- Python 3.11+
- Node.js 18+
- A `GOOGLE_API_KEY` for the Google Search fallback

### Backend setup & run

```bash
cd agentic-langgraph-app/backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
# add GOOGLE_API_KEY to .env
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
cd agentic-langgraph-app/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend calls `http://localhost:8000/api/chat` by default.  
Override with `VITE_API_BASE_URL` in a `.env` file if needed.

---

## Agent graph flow

```
START
  │
  ▼
chatbot ──── no tool needed ──► END
  │
  ▼ (tool call)
tools
  │
  ▼
chatbot ──── no Google needed ──► END
  │
  ▼ (Google requested)
search already done? ── yes ──► chatbot ──► END
  │ no
  ▼
google_search
  │
  ▼
chatbot ──► END
```

---

## LangGraph Course

Jupyter notebooks covering LangGraph fundamentals.

### Setup

```bash
cd langgraph-course
# rename sample.env to .env and add your keys
cp sample.env .env
```

Install dependencies with `uv` or `pip` as described in the course notebooks.
