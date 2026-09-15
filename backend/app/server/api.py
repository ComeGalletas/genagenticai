from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import json
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from ..graph.core.graph import run_agent, stream_agent_events
from ..db.chroma_store import load_vectorstore
from ..server.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = RotatingFileHandler(
    _LOG_DIR / "app.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=3,             # keeps up to 3 old log files
    encoding="utf-8",
)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan. Loads the vector store on startup and handles shutdown."""
    logger.info("Agentic LangGraph API starting up — logs -> %s", _LOG_DIR / "app.log")

    try:
        load_vectorstore()
    except Exception as exc:
        logger.error("Failed to initialise vector store on startup: %s", exc)
        # raise
    # Startup is complete
    yield
    # Shutdown logic goes here
    logger.info("Agentic LangGraph API shutting down")

app = FastAPI(title="Agentic LangGraph API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint: returns a simple status message."""
    logger.debug("Health check requested")
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint: receives a message, invokes the agent, and returns the reply."""
    logger.info(
        "Chat request — thread_id=%r message=%r",
        request.thread_id,
        request.message[:120],
    )
    try:
        start = time.perf_counter()
        reply = run_agent(request.message, thread_id=request.thread_id)
        elapsed = time.perf_counter() - start
        
        logger.info("Chat response in %.2fs — thread_id=%r", elapsed, request.thread_id)
        logger.debug("Chat response content: %r", reply[:120])
        return ChatResponse(reply=reply)
    except Exception as exc:
        logger.error("Chat error — thread_id=%r: %s", request.thread_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _sse(event: dict) -> str:
    """Format one event as a Server-Sent Events frame."""
    return f"event: {event['event']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _sse_stream(request: ChatRequest) -> Iterator[str]:
    start = time.perf_counter()
    for event in stream_agent_events(request.message, thread_id=request.thread_id):
        yield _sse(event)
    logger.info("Chat stream completed in %.2fs — thread_id=%r", time.perf_counter() - start, request.thread_id)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint (Server-Sent Events over a POST).

    Frames: status (progress hint), delta (chatbot tokens, markdown), reset (discard the draft),
    final (sanitized HTML reply), done, or error. The client should replace whatever it has
    drafted with the `final` HTML.
    """
    logger.info("Chat stream request — thread_id=%r message=%r", request.thread_id, request.message[:120])
    return StreamingResponse(
        _sse_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
