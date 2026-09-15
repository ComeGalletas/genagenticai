from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

USE_RETRIEVAL_PIPELINE_TOOL = True  # If True it will use the retrieval pipeline as a tool. Set to False to use it as a node.

# Models. The chatbot runs with thinking enabled; the judge reuses the same model with thinking
# disabled so it only emits a short JSON verdict. Keeping both on ONE model matters on a single
# consumer GPU: qwen3.6 (23 GB) already spills past a 16 GB card, so loading a separate judge model
# evicts the chat model and costs 5-11 s per switch, more than the judge call itself.
# Override JUDGE_MODEL only if both models fit in VRAM together.
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.6")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)
JUDGE_KEEP_ALIVE = os.getenv("JUDGE_KEEP_ALIVE", "10m")  # keep the judge loaded between turns
JUDGE_MAX_RETRIEVALS = int(os.getenv("JUDGE_MAX_RETRIEVALS", "3"))  # most recent retrieval entries shown to the judge
JUDGE_MAX_DOC_CHARS = int(os.getenv("JUDGE_MAX_DOC_CHARS", "1200"))  # per-document cap in the judge prompt

@dataclass(frozen=True)
class Settings:
    google_api_key: str
    model_name: str = "gemini-1.5-flash"

def get_settings() -> Settings:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    print(f"GOOGLE_API_KEY: {api_key}")  # Debugging line to check the value of GOOGLE_API_KEY
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Settings(google_api_key=api_key)
