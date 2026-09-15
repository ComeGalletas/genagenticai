# Docker Journal: containerised frontend + backend

Goal: run the React client and the FastAPI/LangGraph backend as two containers in one Compose
stack, with Ollama staying on the host (that is where the GPU is). A single container for both
was considered and rejected: it would need a process supervisor and two runtimes in one image for
no gain over Compose.

## Decisions

- **Ollama on the host.** The backend reaches it through `host.docker.internal:11434`
  (`OLLAMA_BASE_URL`), with the `host-gateway` extra host so the same file works on Linux.
- **Chroma store in a named volume, rebuilt on first start.** The store is no longer in git
  (`.gitignore` since 2026-09-15). Baking it into the image would need Ollama reachable during
  `docker build`; a volume plus an entrypoint rebuild keeps the build offline.
- **Frontend served by nginx, which proxies `/api/` and `/health` to the backend.** The browser
  talks to one origin, so `VITE_API_BASE_URL` is built empty (relative URLs) and CORS is not
  involved. SSE streaming needs `proxy_buffering off` on the API location.

## Todo

- [x] 1. Backend image `backend/Dockerfile` (python:3.12-slim, uv sync from `uv.lock` without dev,
      copies `app/` and `data/knowledge/`, runs uvicorn on 0.0.0.0:8000) + `backend/.dockerignore`.
- [x] 2. `OLLAMA_BASE_URL` setting passed as `base_url` to both `ChatOllama` instances and
      `OllamaEmbeddings`; Compose sets it to `http://host.docker.internal:11434`.
- [x] 3. Chroma volume on `CHROMA_DIR` + `docker-entrypoint.sh` that runs `python -m app.main rebuild`
      when the store has no ready marker (`REBUILD_ON_START=1` forces it).
- [x] 4. `CORS_ORIGINS` env var replaces the hardcoded Vite origin in `server/api.py`.
- [x] 5. Frontend image `frontend/Dockerfile` (node:20-alpine build, nginx:alpine serve) +
      `frontend/nginx.conf` proxy + `frontend/.dockerignore`.
- [x] 6. `App.jsx`: empty `VITE_API_BASE_URL` means same-origin relative URLs; build arg in the image.
- [x] 7. `docker-compose.yml` at the repo root: backend (env_file `backend/.env` optional, volumes for
      Chroma and logs, `/health` healthcheck), frontend on port 5173 depending on a healthy backend.
- [x] 8. Docker sections in the root, backend and frontend READMEs; `.env.example` documents
      `OLLAMA_BASE_URL` and `CORS_ORIGINS`; `.gitattributes` forces LF on `*.sh`.
- [x] 9. Verify: build both images, `docker compose up`, `/health`, first-start rebuild, a
      search-backed question and a follow-up through the browser at `localhost:5173` (streaming
      through nginx).
- [x] 10. Record image sizes, build times and first-run rebuild time below.

## Log

### 2026-09-15: Docker Desktop would not start (before any image was built)

`docker compose build` failed with "failed to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine". Docker Desktop 4.83.0 (per-user install under
`%LOCALAPPDATA%\Programs\DockerDesktop`) was stopped, and every launch died with:

```
starting services: initializing Inference manager: listening on
unix://C:/Users/<user>/AppData/Local/Docker/run/dockerInference: remove ...: The file cannot be
accessed by the system.
```

The leftover AF_UNIX socket files in `%LOCALAPPDATA%\Docker\run` (and one under
`%LOCALAPPDATA%\docker-secrets-engine`) are reparse points that Windows 11 build 26200 refuses to
open, query or delete (error 1920) even with no Docker process alive; `fsutil reparsepoint delete`
fails the same way. Docker cannot remove them, so its backend aborts before the engine starts.

Workaround that worked: rename the folder aside (renaming the directory is allowed even though the
files inside are not), then start Docker Desktop once and let it recreate the folder. The engine was
up 5 s later. Force-killing Docker Desktop leaves fresh sockets behind and reproduces the failure,
so quit it cleanly. A reboot should clear the stale reparse points and the `*.stale-*` folders can
then be deleted.

```powershell
Rename-Item "$env:LOCALAPPDATA\Docker\run" "run.stale-$(Get-Date -Format yyyyMMddHHmmss)"
Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
```

### 2026-09-15: stack built and verified

| Item | Result |
|---|---|
| `docker compose build` (cold, both images) | 47 s |
| `genagenticai-backend` image | 300 MB (python:3.12-slim + uv sync, no dev group) |
| `genagenticai-frontend` image | 74 MB (nginx:1.27-alpine + Vite `dist/`) |
| First start: Chroma rebuild inside the container | 12 s (knowledge + baloto collections, embeddings from the host's Ollama) |
| Backend healthy after | 5 s once uvicorn was up |
| Container -> host Ollama (`host.docker.internal:11434`) | works with Ollama bound to 127.0.0.1 only; Docker Desktop forwards to the host loopback |

Browser check at `http://localhost:5173` (served by nginx): "How much VRAM does the RTX 5090 have?"
streamed token by token through the proxy and rendered the final HTML; "And what is its TDP?"
re-searched `RTX 5090 TDP` (turn scoping from the rework journal works unchanged in the container).
No console errors. Raw SSE through nginx shows `delta` frames arriving every second, so
`proxy_buffering off` does its job.

Performance investigation. The first container turns took 11 to 19 s against 2.5 to 3 s natively, so
I chased it:

- Raw Ollama streaming from inside a container: 100 chunks/s, first chunk 0.04 s (same as host).
- `ChatOllama.stream()` from inside a container: 104 chunks/s (same as host).
- The benchmark harness and `graph.stream` in messages mode, run inside the container: 2.5 to 2.7 s
  per turn at ~100 tok/s, identical to native.
- Package versions in the image match the native venv exactly (starlette 1.3.1, anyio 4.14.2,
  uvicorn 0.51.0, langgraph 1.2.9, langchain-ollama 0.3.10, ollama 0.6.2, chromadb 1.5.9).
- Ollama's own `server.log` gave the answer: from 13:53:15 every request, whatever the client,
  generated at 22 tok/s instead of ~100, and `nvidia-smi` showed a game (`ffxiv_dx11.exe`) on the
  same GPU with 14.4 of 16.3 GB VRAM in use. The container path is not slower; the GPU was shared.

Judge these numbers with the GPU idle. `ollama ps` still reported "100% GPU" during the slowdown, so
it is not a reliable signal for contention; the `eval time ... tokens per second` lines in
`%LOCALAPPDATA%\Ollama\server.log` are.

Notes:
- The uvicorn access log prints one line per healthcheck; the interval is 30 s to keep
  `compose logs backend` readable.
- Pressing Enter in the chat box does not send (the form only submits from the button); unchanged,
  pre-existing frontend behaviour.
- Conversation memory is the in-process `MemorySaver`, so a container restart forgets threads.
  Unchanged from native.
