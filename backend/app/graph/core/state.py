
from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from langgraph.graph.message import add_messages

from ..retrieval.state import RetrievalState

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_id: NotRequired[str]
    # For retrieval tool
    retrieval: NotRequired[RetrievalState | None]
