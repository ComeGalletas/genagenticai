from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import USE_RETRIEVAL_PIPELINE_TOOL
from .state import State
from ..retrieval.retrieval_service import retrieval_engine


def route_chatbot(state: State):
    """ Decide where to go after the chatbot.
        It is used to redirect to a retrieve_information node instead of the tool option.
    """
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END

    # This can be refactored slighly to allow for this router to also call the respective tool instead of the node
    tool_name = last.tool_calls[0]["name"]
    if not USE_RETRIEVAL_PIPELINE_TOOL and tool_name == "retrieve_information":
        return "retrieve_information"

    return "tools"


def retrieval_router(state: State):
    """ Decide where to go after the retrieve_information node. Returns the name of the node to go next.
        It is used to redirect to a finish_retrieval node instead of the retrieve_information node 
        if the retrieval process is complete or if enough information has been gathered.
    """

    if state.get("retrieval_stage") is None:
        return "finish_retrieval"

    if retrieval_engine._enough_information(
        state.get("retrieval_results", [])
    ):
        return "finish_retrieval"

    return "retrieve_information"