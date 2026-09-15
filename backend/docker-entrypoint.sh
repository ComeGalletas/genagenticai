#!/bin/sh
# Rebuild the Chroma collections on first start (the store is not shipped in the image) or when
# REBUILD_ON_START=1. Needs Ollama reachable at OLLAMA_BASE_URL with the embedding model pulled.
set -e

CHROMA_DIR="${CHROMA_DIR:-/app/data/chroma_db}"
MARKER="$CHROMA_DIR/.ready"

if [ "$1" = "uvicorn" ]; then
    if [ "${REBUILD_ON_START:-0}" = "1" ] || [ ! -f "$MARKER" ]; then
        echo "[entrypoint] Chroma store not ready at $CHROMA_DIR; rebuilding from $KNOWLEDGE_DIR via Ollama at ${OLLAMA_BASE_URL:-localhost:11434} ..."
        python -m app.main rebuild
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
        echo "[entrypoint] Chroma store ready."
    else
        echo "[entrypoint] Chroma store ready since $(cat "$MARKER"); set REBUILD_ON_START=1 to rebuild."
    fi
fi

exec "$@"
