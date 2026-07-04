from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import State


def retrieval_router(state: State):

    pipeline = state.get("pending_pipeline")
    if not pipeline:
        return "chatbot"

    return pipeline[0]
