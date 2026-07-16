from typing_extensions import TypedDict, NotRequired

from ...retrieval.schemas import RetrievalResult


class RetrievalState(TypedDict):
    """State used only by the retrieval functionalities."""
    messages: list
    retrieval_query: NotRequired[str]
    retrieval_stage: NotRequired[int | None]
    retrieval_results: NotRequired[list[RetrievalResult]]
    tool_call_id: NotRequired[str]