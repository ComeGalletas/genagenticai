# Backend (LangGraph + FastAPI)

## 1) Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
```

Requires [Ollama](https://ollama.com/) running locally with the chat model and the embedding model pulled
(defaults: `granite4.1:8b` and `nomic-embed-text`). Models and tuning knobs are set in `.env`; see
`.env.example` and `app/config.py`.

## 2) Run

```bash
uvicorn app.server.api:app --reload --reload-dir app --port 8000
```

Health check: `http://localhost:8000/health`

Endpoints:

- `POST /api/chat` — one-shot: `{message, thread_id}` -> `{reply}` (sanitized HTML)
- `POST /api/chat/stream` — same body, Server-Sent Events: `status`, `delta`, `reset`, `final`, `done`/`error`

## 3) Commands

```bash
python -m app.main rebuild          # rebuild the Chroma collections (knowledge, baloto)
python -m app.main graph            # render the agent graph to graph.png
python -m unittest discover tests   # unit tests (no Ollama needed)
python -m benchmarks.run_benchmark  # fixed question set with per-node timings (needs Ollama)
```

## 4) Graph

```
START -> prepare_turn -> chatbot --tool calls--> tools --> chatbot
                            |
                       final answer
                            v
                 should_judge? (tools used or >= 300 chars)
                    no |                 | yes
                       v                 v
                   finalize            judge --pass--> finalize -> END
                                         | fail, one retry
                                         v
                                       revise -> chatbot
```

- `prepare_turn`: language detection, per-turn state reset.
- `chatbot`: Ollama chat model with tools (`retrieve_information`, `retrieve_baloto_results`,
  `suggest_baloto_numbers`, `retrieve_job_postings`, `read_webpage`, `get_current_time`).
- `retrieve_information` runs static index -> local RAG (Chroma) -> web (DuckDuckGo) and stops at the
  first decisive result.
- `judge`: same model, thinking off, structured verdict (passed / grounded / needs_more_info / issues).
- `revise`: feeds the verdict back to the chatbot once.
- `finalize`: markdown -> sanitized HTML.

See `../rework_journal.md` for design notes and measurements.
