import logging
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.types import Command
from langgraph.graph import END

from .state import State
from .tools import retrieval_engine, format_results, retrieve_information, get_current_time, static_google_search

from .system_prompt import SYSTEM_PROMPT
SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)

logger = logging.getLogger(__name__)
tools = [static_google_search, get_current_time, retrieve_information]

_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
llm_with_tools = _llm.bind_tools(tools)


def chatbot(state: State) -> Command:
    """Invoke the LLM and append its response to the conversation."""
    logger.debug("Entering chatbot node.")
    
    start_time = time.perf_counter()
    response = llm_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])
    elapsed = time.perf_counter() - start_time

    logger.info("LLM response completed in %.3f seconds.", elapsed)

    if response.tool_calls:
        logger.info("LLM requested tool: %s", response.tool_calls[0]["name"])
    else:
        logger.info("LLM produced final response.")

    return Command(update={"messages": [response]})


def retrieve_information_node(state: State):
    """ Retrieves information from a knowledge base and web search functions. It includes a knowledge base and web search functions.
        It is a very very slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        Use it if you need to know the answer to a question that requires external knowledge or recent information.
    """
    logger.info("Retrieve_information node called")
    # First time entering the node
    if state.get("retrieval_query") is None:
        ai_message = state["messages"][-1]
        tool_call = ai_message.tool_calls[0]
        query = tool_call["args"]["query"]
        tool_call_id = tool_call["id"]

        stage = 0
        accumulated_results = []

    # Returning from a previous stage
    else:
        query = state.get("retrieval_query")
        stage = state.get("retrieval_stage")
        accumulated_results = state.get("retrieval_results")
        tool_call_id = state.get("tool_call_id")

    if stage is None:
        raise RuntimeError("retrieve_information_node called after retrieval completed")
    stage_results, next_stage = retrieval_engine.run_stage(query=str(query), stage=stage)
    if accumulated_results is not None:
        accumulated_results.extend(stage_results)

    return Command(
        update={
            "retrieval_query": query,
            "retrieval_stage": next_stage,
            "retrieval_results": accumulated_results,
            "tool_call_id": tool_call_id,
        }
    )


def finish_retrieval_node(state: State):
    """Finish the retrieval process and format the results then go back to the chatbot node."""
    formatted = format_results(
        query=str(state.get("retrieval_query")),
        results=state.get("retrieval_results", []),
    )
    return Command(update={"messages": [ToolMessage(content=formatted, tool_call_id=state.get("tool_call_id"))
            ],
            "retrieval_query": None,
            "retrieval_stage": None,
            "retrieval_results": [],
            "tool_call_id": None,
        }
    )
