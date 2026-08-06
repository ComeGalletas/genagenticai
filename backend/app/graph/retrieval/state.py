from typing import Any, Literal

from attr import field
from typing_extensions import TypedDict, NotRequired

from ...retrieval.schemas import RetrievalResult

RetrievalStatus = Literal[
    "NO_MATCH",
    "FOUND"
]

class RetrievalDocument(TypedDict):
    title: str
    content: str
    source: str
    stage: str
    confidence: float | None
    metadata: dict[str, Any]


class RetrievalState(TypedDict):
    """State used only by the retrieval functionalities."""
    retrieval_query: NotRequired[str]
    # Control
    retrieval_status: RetrievalStatus | None
    retrieval_stage: NotRequired[int | None]
    retrieval_next_stage: NotRequired[int | None]
    # Results
    retrieval_documents: NotRequired[list[RetrievalDocument | None]]