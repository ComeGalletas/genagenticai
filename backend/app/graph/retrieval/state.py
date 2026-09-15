from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

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
    # Id of the tool call that produced this entry; ties the entry to a conversation turn.
    tool_call_id: NotRequired[str]
    # Control
    retrieval_status: RetrievalStatus | None
    retrieval_stage: NotRequired[int | None]
    retrieval_next_stage: NotRequired[int | None]
    # Results
    retrieval_documents: NotRequired[list[RetrievalDocument | None]]