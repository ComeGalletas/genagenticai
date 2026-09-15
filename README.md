# Generative Agentic AI

A learning-focused agentic AI workspace built with **LangGraph**, **FastAPI**, and **React**. The agent uses a locally running **Ollama** LLM (granite4.1:8b by default, any tool-capable model works) and decides at runtime which tool to invoke to answer a question:

- **Staged retrieval** — one tool call searches a curated in-code index, then a local **ChromaDB** knowledge base (Ollama embeddings), then the live web via **DuckDuckGo**, stopping at the first decisive result
- **Baloto lottery** — draw history from a dedicated Chroma collection plus hot/cold/hybrid number suggestions
- **Job Posting Search** — recent LinkedIn postings by query, location and remote option
- **Web page reader** and **current time**

Every tool-backed answer is checked by a **judge** pass (same model, structured verdict) and sent back for one revision if it is ungrounded, then rendered from markdown to sanitized HTML and **streamed** to the React client. No paid cloud LLMs involved.

## Project structure

- `backend/`: FastAPI + LangGraph agent
- `frontend/`: React chat client

## Agentic LangGraph App

### Prerequisites

- [Ollama](https://ollama.com/) running locally with `granite4.1:8b` and `nomic-embed-text` pulled (or set `CHAT_MODEL` to another tool-capable model)
- Python 3.12+
- Node.js 18+

### Backend setup & run

```bash
cd backend
python -m venv .venv

.venv\Scripts\activate # Optional

pip install -e .
copy .env.example .env
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

Open `http://localhost:5173`. The frontend streams replies from `http://localhost:8000/api/chat/stream`
(Server-Sent Events) and falls back to the one-shot `POST /api/chat` if streaming is unavailable.
Override the base URL with `VITE_API_BASE_URL` in a `.env` file if needed.

Models are configured in `backend/.env` (`CHAT_MODEL`, `JUDGE_MODEL`); see `rework_journal.md` for the
current architecture, measurements, and the benchmark harness in `backend/benchmarks/`.

### Docker (frontend + backend, Ollama on the host)

Prerequisites: Docker Desktop, and Ollama running on the host with the chat and embedding models
pulled (`ollama pull granite4.1:8b`, `ollama pull nomic-embed-text`).

```bash
docker compose up --build
```

Open `http://localhost:5173`. nginx serves the built client and proxies `/api/` and `/health` to the
backend container, which reaches Ollama at `host.docker.internal:11434`. The first start rebuilds the
Chroma store into a named volume (a minute or two of embedding); later starts reuse it.

```bash
docker compose logs -f backend                 # agent logs
REBUILD_ON_START=1 docker compose up backend   # force a store rebuild after editing data/knowledge
docker compose down -v                         # stop and drop the Chroma and log volumes
```

Settings still come from `backend/.env`; `docker-compose.yml` only overrides the Ollama URL, CORS
origins and data paths. See `docker_journal.md` for the design notes.

---

