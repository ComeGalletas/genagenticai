from enum import StrEnum
from dataclasses import dataclass, field
from typing import Any

"""Just names, not function references."""
class RetrievalStage(StrEnum):
    STATIC = "query_static_search"
    RAG = "query_knowledge"
    GOOGLE = "query_ddu_google_search"


@dataclass(slots=True)
class RetrievalResult():
    title: str
    content: str
    source: str
    stage: RetrievalStage
    #status: str          # "FOUND", "NO_MATCH", "ERROR"
    score: float |None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
