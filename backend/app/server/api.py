from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..graph.graph import run_agent
from ..db.chroma_store import load_vectorstore
from ..server.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).parent.parent / "logs"
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
