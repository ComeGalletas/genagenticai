from langgraph.graph import END

from .state import State
from typing import cast

from ...retrieval.service import retrieval_engine

from langchain_core.messages import AIMessage


from ...config import USE_RETRIEVAL_PIPELINE_TOOL # to be deprecated in the future, it is used to disable the retrieval pipeline and use the tools instead.


def route_chatbot(state: State):
    """ Decide where to go after the chatbot.
        It is used to redirect to a retrieve_information node instead of the tool option.
    """
    # If the last message from the LLM does not contain tool calls,
    # route to judge so the final response is polished before returning.
    last = state["messages"][-1]

    print("Last state:", state)
    
    if not getattr(last, "tool_calls", None):
        print("Routing to judge")
        return "judge"

    # This can be refactored to allow for more complex routing
    tool_name = cast(AIMessage, last).tool_calls[0]["name"]
    if not USE_RETRIEVAL_PIPELINE_TOOL and tool_name == "retrieve_information":
        print("Routing to retrieve_information")
        return "retrieve_information"

    return "tools"


def retrieval_router(state: State):
    """ Decide where to go after the retrieve_information node. Returns the name of the node to go next.
        It is used to redirect to a finish_retrieval node instead of the retrieve_information node 
        if the retrieval process is complete or if enough information has been gathered.
    """
    # If the retrieval stage is None, it means that the retrieval process is complete and we can go to the finish_retrieval node.
    if state.get("retrieval_stage") is None:
        return "finish_retrieval"

    # If the retrieval results are enough, the stages gets clear and gets routed to the finish_retrieval node.
    if retrieval_engine._enough_information(state.get("retrieval_results", [])):
        return "finish_retrieval"

    return "retrieve_information"