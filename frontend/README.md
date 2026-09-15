# Frontend (React + Vite)

## 1) Setup

```bash
cd frontend
npm install
```

## 2) Run

```bash
npm run dev
```

By default, the app calls `http://localhost:8000/api/chat`.
Use `VITE_API_BASE_URL` in a `.env` file if needed.

## 3) Docker

`Dockerfile` builds the app with `VITE_API_BASE_URL=""` (same-origin relative URLs) and serves `dist/`
with nginx; `nginx.conf` proxies `/api/` and `/health` to the `backend` service with buffering off so
streamed replies arrive token by token. Run the whole stack from the repo root with
`docker compose up --build` and open `http://localhost:5173`.
