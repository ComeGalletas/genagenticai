from dataclasses import dataclass, field
from typing import Any


"""Just names, not function references for the available retrieval functions metadata."""
class RetrievalStage:
    STATIC: str = "query_static_search"
    RAG: str = "query_knowledge"
    WEB: str = "query_web_search"


"""Defines the structure of a retrieval result for LLM usage, including title, content, source URL, stage of retrieval, status, score, confidence, and additional metadata."""
@dataclass(slots=True)
class RetrievalResult():
    title: str
    content: str
    source: str
    stage: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
