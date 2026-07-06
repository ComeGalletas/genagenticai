from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

USE_RETRIEVAL_PIPELINE = True  # Set to False to disable the retrieval pipeline

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
