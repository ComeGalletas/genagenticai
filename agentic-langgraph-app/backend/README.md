# Backend (LangGraph)

## 1) Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
```

Add your `GOOGLE_API_KEY` to `.env`.

## 2) Run

```bash
uvicorn app.main:app --reload --port 8000
```

Test at `http://localhost:8000/health`.
