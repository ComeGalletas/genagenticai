from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_BACKEND_DIR = Path(__file__).resolve().parents[2]

# Every value falls back to the literal below when the env var is absent.
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or _BACKEND_DIR / "data" / "chroma_db")
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR") or _BACKEND_DIR / "data" / "knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_COLLECTION = os.getenv("CHROMA_DEFAULT_COLLECTION", "knowledge")

RAG_K = int(os.getenv("RAG_K", "3"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.44"))
RAG_ROUTING_MIN_SCORE = int(os.getenv("RAG_ROUTING_MIN_SCORE", "3"))
