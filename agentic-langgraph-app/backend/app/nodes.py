import logging
from typing import Annotated, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.graph import END

from .rag import query_knowledge
from .state import State
from .tools import STATIC_RESULTS, format_results, static_to_results, ddu_google_search, static_google_search, retrieve_information

SYSTEM_PROMPT = """You are a helpful assistant with access to a local knowledge base and a search tool.
    Rules you must always follow:
    1. For EVERY user question, call knowledge_search first to check the local knowledge base before answering.
    2. Only call static_google_search if knowledge_search returned no useful result. ALWAYS.
    3. Never answer from memory alone when a tool could provide a better answer.
    4. Extract all proper nouns and identifiers exactly as written — do not modify them.
    5. If there are no tools or options available, answer the question to the best of your ability using your own knowledge.
    6. Add "nya" before any comma or period, like an anime character.
"""
SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)


logger = logging.getLogger(__name__)

tools = [static_google_search]

_llm = ChatOllama(
    model="qwen3",
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

    logger.info("LLM produced final response.")
    return Command(goto=END, update={"messages": [response]})


def static_search(state: State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    query = state.get("retrieval_query")
    normalized = query.lower().strip() if query else ""

    results: list[Dict[str, str]] = []    
    for entry in STATIC_RESULTS:
        keywords = entry["keywords"].split()
        if any(word.lower() in normalized for word in keywords):
            results.append({
                "query": str(query),
                "title": entry["title"],
                "url": entry["url"],
                "snippet": entry["snippet"],
            })

    retrieval_results = static_to_results(results)

    tool_message = format_results(query=str(query), results=retrieval_results)
    return Command(update={"messages": [ToolMessage(content=tool_message, tool_call_id=tool_call_id)]})


def rag_search(state: State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search the local knowledge base. Use this tool for ANY question before answering.
    The knowledge base covers: NVIDIA RTX 50 series GPUs (RTX 5090, 5080, 5070 Ti, 5070,
    5060 Ti, 5060, 5050), their specs, features, performance, and pricing. Also covers
    LangGraph, RAG pipelines, ChromaDB, Ollama, FastAPI, and this project's architecture."""
    query = state.get("retrieval_query")
    logger.info("Knowledge search tool called — question: %r", str(query))

    retrieval_results = query_knowledge(str(query))

    tool_message = format_results(query=str(query), results=retrieval_results)
    return Command(update={"messages": [ToolMessage(content=tool_message, tool_call_id=tool_call_id)]})


def google_duck_search(state: State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    query = state.get("retrieval_query")

    results = ddu_google_search(str(query))

    return Command(update={"messages": [ToolMessage(content=format_results(query=str(query), results=results), tool_call_id=tool_call_id)]})

