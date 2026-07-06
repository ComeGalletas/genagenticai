import logging
from typing import Annotated, Dict
from urllib import response

from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.graph import END

from .state import State
from .tools import format_results, retrieve_information, retrieval_engine, get_current_time
from ..db.rag import query_knowledge
from ..searches.static_search import static_google_search, query_static_search
from ..searches.google_search import query_ddu_google_search

from .system_prompt import SYSTEM_PROMPT
SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)

logger = logging.getLogger(__name__)
tools = [static_google_search, get_current_time, retrieve_information]

_llm = ChatOllama(
    model="qwen3.6",
    temperature=0,
)
llm_with_tools = _llm.bind_tools(tools)


def chatbot(state: State) -> Command:
    """Invoke the LLM and append its response to the conversation."""
    logger.debug("Entering chatbot node.")
    response = llm_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])
    logger.debug("LLM response received.")

    if response.tool_calls:
        logger.info("LLM requested tool: %s", response.tool_calls[0]["name"])
    else:
        logger.info("LLM produced final response.")

    return Command(update={"messages": [response]})

def retrieve_information_node(state: State):

    ai_message = state["messages"][-1]
    tool_call = ai_message.tool_calls[0]

    query = tool_call["args"]["query"]
    tool_call_id = tool_call["id"]

    results = retrieval_engine.run_pipeline(query)
    formatted = format_results(
        query=query,
        results=results,
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=formatted,
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )
