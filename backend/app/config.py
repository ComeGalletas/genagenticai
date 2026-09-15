"""Application settings. Every value can be overridden from the environment or backend/.env.

See backend/.env.example for the ones you are most likely to change.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Retrieval pipeline mode: True runs it inside the `retrieve_information` tool (default);
# False wires it as graph nodes (`retrieve_information` -> `finish_retrieval`).
USE_RETRIEVAL_PIPELINE_TOOL = os.getenv("USE_RETRIEVAL_PIPELINE_TOOL", "true").lower() not in {"0", "false", "no"}

# ---------------------------------------------------------------------------
# Models (Ollama)
# ---------------------------------------------------------------------------
# The chatbot runs with thinking enabled; the judge reuses the same model with thinking disabled
# so it only emits a short JSON verdict. Keeping both on ONE model matters on a single consumer
# GPU: a model that does not fit in VRAM gets evicted on every chat/judge switch, which costs
# more than the judge call itself. Override JUDGE_MODEL only if both models fit together.
CHAT_MODEL = os.getenv("CHAT_MODEL", "granite4.1:8b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)
CHAT_KEEP_ALIVE = os.getenv("CHAT_KEEP_ALIVE", "10m")
JUDGE_KEEP_ALIVE = os.getenv("JUDGE_KEEP_ALIVE", "10m")

# Context budget. num_ctx sizes the KV cache Ollama allocates, so keep it close to what is used:
# system prompt (~1.2k tokens) + retrieval context (bounded below) + trimmed history + reply.
CHAT_NUM_CTX = int(os.getenv("CHAT_NUM_CTX", "16384"))
# Ollama reloads a model whenever num_ctx changes between requests, so when the judge shares the
# chat model it must use the same context size or every chat/judge switch pays a reload.
JUDGE_NUM_CTX = int(os.getenv("JUDGE_NUM_CTX", str(CHAT_NUM_CTX if JUDGE_MODEL == CHAT_MODEL else 8192)))
CHAT_HISTORY_MAX_TOKENS = int(os.getenv("CHAT_HISTORY_MAX_TOKENS", "6000"))  # approximate tokens of past messages sent to the chatbot

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_MAX_ENTRIES = int(os.getenv("RETRIEVAL_MAX_ENTRIES", "3"))  # newest retrieval tool results kept in state (0 = unbounded)
# One tool call runs the stages in order (static index -> local RAG -> web) and stops at the first
# stage that yields at least RETRIEVAL_STOP_MIN_RESULTS *decisive* results.
RETRIEVAL_STOP_MIN_RESULTS = int(os.getenv("RETRIEVAL_STOP_MIN_RESULTS", "1"))
RETRIEVAL_CACHE_TTL = float(os.getenv("RETRIEVAL_CACHE_TTL", "300"))  # seconds a (query, stage) result is reused; 0 disables
RETRIEVAL_CACHE_SIZE = int(os.getenv("RETRIEVAL_CACHE_SIZE", "128"))
WEB_FETCH_TIMEOUT = float(os.getenv("WEB_FETCH_TIMEOUT", "5"))  # per-page HTTP timeout for search result enrichment
WEB_FETCH_WORKERS = int(os.getenv("WEB_FETCH_WORKERS", "5"))  # pages fetched concurrently

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_MAX_RETRIEVALS = int(os.getenv("JUDGE_MAX_RETRIEVALS", "3"))  # most recent retrieval entries shown to the judge
JUDGE_MAX_DOC_CHARS = int(os.getenv("JUDGE_MAX_DOC_CHARS", "1200"))  # per-document cap in the judge prompt
JUDGE_MAX_TOOL_CHARS = int(os.getenv("JUDGE_MAX_TOOL_CHARS", "8000"))  # per tool-output cap; job lists and lottery JSON are large
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "1"))  # how many times a failed answer is sent back for revision per turn
JUDGE_SKIP_MAX_CHARS = int(os.getenv("JUDGE_SKIP_MAX_CHARS", "300"))  # replies shorter than this that used no tools skip the judge

# Link verification: before judging, fetch the pages behind links the answer cites (and behind
# curated index entries it relied on) so the judge can check that they support the claims.
# Links that come from tool outputs or already-fetched web documents are trusted and not re-fetched.
JUDGE_VERIFY_LINKS = os.getenv("JUDGE_VERIFY_LINKS", "true").lower() not in {"0", "false", "no"}
JUDGE_MAX_VERIFY_LINKS = int(os.getenv("JUDGE_MAX_VERIFY_LINKS", "5"))  # pages fetched per answer
JUDGE_VERIFY_TIMEOUT = float(os.getenv("JUDGE_VERIFY_TIMEOUT", "5"))  # seconds per page
JUDGE_VERIFY_PAGE_CHARS = int(os.getenv("JUDGE_VERIFY_PAGE_CHARS", "2000"))  # excerpt shown to the judge per page
