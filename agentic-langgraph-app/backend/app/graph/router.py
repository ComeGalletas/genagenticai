from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import State
from ..config import USE_RETRIEVAL_PIPELINE


def route_chatbot(state: State):
    """Decide where to go after the chatbot."""

    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        return END

    tool_name = last.tool_calls[0]["name"]
    if USE_RETRIEVAL_PIPELINE and tool_name == "retrieve_information":
        return "retrieve_information"

    return "tools"
