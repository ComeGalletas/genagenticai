import logging
from typing import Annotated, Dict
from urllib import response

from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.graph import END

from .state import State
from .searches.rag import query_knowledge
from .searches.static_search import static_google_search, query_static_search
from .searches.google_search import query_ddu_google_search
from .tools import format_results, retrieve_information

SYSTEM_PROMPT = """You are a helpful assistant with access to a local knowledge base and a search tool.
Add "nya" before any comma or period, like an anime character.
"""
SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)


logger = logging.getLogger(__name__)

tools = [static_google_search, retrieve_information]

_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
llm_with_tools = _llm.bind_tools(tools)


def chatbot(state: State) -> Command:
    """Main LLM node: invoke the LLM, append the response, and route to the next node."""
    logger.debug("Entering chatbot node.")

    response = llm_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])
    logger.debug("LLM response received.")

    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        logger.info("LLM requested tool: %s", tool_name)

        if tool_name == "retrieve_information":
            return Command(goto="initialize_retrieval", update={"messages": [response]})

        if tool_name == "continue_retrieval":
            pipeline = state.get("retrieval_pipeline", [])
            index = state.get("retrieval_index", 0)
            if index < len(pipeline):
                next_node = pipeline[index]
                logger.info("Continuing retrieval -> %s", next_node)
                return Command(goto=next_node, update={"messages": [response]})
            logger.info("Retrieval pipeline finished.")
            return Command(goto=END, update={"messages": [response]})

        return Command(goto="tools", update={"messages": [response]})

    logger.info("%r", response)
    logger.info("%r", response.additional_kwargs)
    logger.info("%r", response.response_metadata)
    logger.info("%r", response.tool_calls)
    logger.info("LLM produced final response.")
    return Command(goto=END, update={"messages": [response]})


def static_search(state: State) -> Command:
    """Search the static google search. """
    query = state.get("retrieval_query")
    logger.info("Static search tool called — question: %r", str(query))

    retrieval_results = query_static_search(str(query))
    tool_message = format_results(query=str(query), results=retrieval_results)
    return Command(update={"messages": [ToolMessage(content=tool_message)]})

def rag_search(state: State) -> Command:
    """Search the local knowledge base. Use this tool for ANY question before answering.
    The knowledge base covers: NVIDIA RTX 50 series GPUs (RTX 5090, 5080, 5070 Ti, 5070,
    5060 Ti, 5060, 5050), their specs, features, performance, and pricing. Also covers
    LangGraph, RAG pipelines, ChromaDB, Ollama, FastAPI, and this project's architecture."""
    query = state.get("retrieval_query")
    logger.info("Knowledge search tool called — question: %r", str(query))

    retrieval_results = query_knowledge(str(query))
    tool_message = format_results(query=str(query), results=retrieval_results)
    return Command(update={"messages": [ToolMessage(content=tool_message)]})

def google_duck_search(state: State) -> Command:
    """Search Google through DuckDuckGo and return structured retrieval results."""
    query = state.get("retrieval_query")
    logger.info("Google DuckDuckGo search tool called — question: %r", str(query))

    retrieval_results = query_ddu_google_search(str(query))
    tool_message = format_results(query=str(query), results=retrieval_results)
    return Command(update={"messages": [ToolMessage(content=tool_message)]})

