import logging
import time
from typing import cast

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command

from .state import State
from .tools import get_current_time, read_webpage, retrieve_information, retrieve_job_postings
from .system_prompt import SYSTEM_PROMPT

SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)

logger = logging.getLogger(__name__)

tools = [get_current_time, read_webpage, retrieve_information, retrieve_job_postings]
_llm = ChatOllama(
    model="qwen3.6",
    #model="qwen3:14b", 
    #model="qwen3:30b-a3b",
    temperature=0.1
)
llm_with_tools = _llm.bind_tools(tools)


def chatbot(state: State) -> Command:
    """Invoke the LLM and append its response to the conversation."""
    logger.debug("Entering chatbot node.")
    
    start_time = time.perf_counter()
    response = cast(AIMessage, llm_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]]))
    elapsed = time.perf_counter() - start_time

    logger.info("LLM response completed in %.3f seconds.", elapsed)
    logger.info("State stage:  %s", state.get("messages"))

    if response.tool_calls:
        logger.info("LLM requested tool: %s", response.tool_calls[0]["name"])
        #print("LLM response:", response)
    else:
        logger.info("LLM produced final response.")
        #print("LLM response:", response)

    return Command(update={"messages": [response]})



