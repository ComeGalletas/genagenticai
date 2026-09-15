# Backend image: FastAPI + LangGraph agent. Ollama is NOT included; it runs on the host and is
# reached through OLLAMA_BASE_URL (see docker-compose.yml). The Chroma store lives in a volume and
# is rebuilt on first start by docker-entrypoint.sh.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CHROMA_DIR=/app/data/chroma_db \
    KNOWLEDGE_DIR=/app/data/knowledge

WORKDIR /app

# uv resolves from uv.lock so the image matches the developer environment exactly.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

# Dependencies first so code changes do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY data/knowledge ./data/knowledge
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p /app/logs /app/data/chroma_db

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.server.api:app", "--host", "0.0.0.0", "--port", "8000"]
