
import operator
from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import MessagesState

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ..retrieval.state import RetrievalState
from ..judge.state import JudgeState

class State(MessagesState):
    # messages: Annotated[list[AnyMessage], add_messages] if not using messagesstate
    user_language: NotRequired[str | None]
    tool_call_id: NotRequired[str]
    # For retrieval tool
    retrieval: NotRequired[Annotated[list[RetrievalState], operator.add]]
    judge: NotRequired[JudgeState | None]
