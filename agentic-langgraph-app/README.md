# Agentic AI Starter (LangGraph + React)

This project includes:
- A LangGraph backend agent with one tool: `static_google_search`
- A React frontend chat UI to talk to the backend

## Project structure

- `backend/`: FastAPI + LangGraph agent
- `frontend/`: React chat client

## Quick start

1. Start backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
# add GOOGLE_API_KEY to .env
uvicorn app.main:app --reload --port 8000
```

2. Start frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Notes

The search tool is intentionally static and deterministic for learning/demo use.
It returns the first matching result from a local in-code index.
