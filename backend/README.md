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

## 4) Docker

`Dockerfile` builds a python:3.12-slim image from `uv.lock` (no dev group) and runs uvicorn on port
8000. `docker-entrypoint.sh` rebuilds the Chroma collections into `CHROMA_DIR` on first start (or when
`REBUILD_ON_START=1`) and writes a `.ready` marker. Ollama stays on the host: `OLLAMA_BASE_URL` is set
to `http://host.docker.internal:11434` by the root `docker-compose.yml`. Run the stack from the repo
root with `docker compose up --build`; see the root README and `docker_journal.md`.

## 5) Graph

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
