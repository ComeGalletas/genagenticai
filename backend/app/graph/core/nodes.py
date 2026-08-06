from dataclasses import asdict
import logging
import time
from typing import cast
from langdetect import detect


from ..judge import state
from ...retrieval.schemas import RetrievalResult
# from attrs import asdict

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from .state import State
from .tools import get_current_time, read_webpage, retrieve_information, retrieve_job_postings
from .system_prompt import SYSTEM_PROMPT
from ..judge.system_prompt import JUDGE_SYSTEM_PROMPT

SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)
JUDGE_SYSTEM_MESSAGE = SystemMessage(content=JUDGE_SYSTEM_PROMPT)

logger = logging.getLogger(__name__)

tools = [get_current_time, read_webpage, retrieve_information, retrieve_job_postings]
_llm = ChatOllama(
    #model="qwen3.6",
    model="qwen3:14b", 
    #model="qwen3:30b-a3b",
    temperature=0.1
)
llm_with_tools = _llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return str(content)


def build_retrieval_context(state: State) -> SystemMessage | None:
    """Converts the retrieval state into a temporary system message."""
    retrievals = state.get("retrieval", [])
    if not retrievals:
        return None

    parts = [
        "## Verified Retrieval Context",
        "",
        "Use this information as the primary factual context when answering.",
        "If the retrieved information is sufficient, prefer it over your internal knowledge.",
        "",
    ]

    for retrieval in retrievals:
        if retrieval["retrieval_status"] != "FOUND":
            continue

        query = retrieval.get("retrieval_query", "")
        parts.append(f"### Retrieval Query")
        parts.append(query)
        parts.append("")


        for doc in retrieval.get("retrieval_documents", []):
            if doc is not None:
                parts.append(f"Source: {doc['source']}")
                parts.append(f"Title: {doc['title']}")
                parts.append(f"Stage: {doc['stage']}")
                parts.append(f"Confidence: {doc['confidence']}")
                parts.append("")
                parts.append(doc['content'])
                parts.append("")
                parts.append("-" * 40)
                parts.append("")

    if len(parts) < 1:
        return None

    return SystemMessage(content="\n".join(parts))

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def chatbot(state: State) -> Command:
    """Invoke the LLM and append its response to the conversation.
    Args:
        state: The current state of the conversation, including messages and any relevant context.
    Returns:
        A Command object for the LLM containing the updated conversation state with the LLMs response.
    """
    logger.debug("Entering chatbot node.")

    messages: list = []
    retrieval_context = build_retrieval_context(state)
    if retrieval_context:
        messages.append(retrieval_context)

    messages.extend(state["messages"])

    user_message = state["messages"][-1].content
    language = detect(user_message)
    state["user_language"] = language
    
    start_time = time.perf_counter()
    response = cast(AIMessage, llm_with_tools.invoke([SYSTEM_MESSAGE, *messages]))
    elapsed = time.perf_counter() - start_time

    logger.info("LLM response completed in %.3f seconds.", elapsed)
    if response.tool_calls:
        logger.info("LLM requested tool: %s", response.tool_calls[0]["name"])
    else:
        logger.info("LLM produced final response.")
        
    #print("LLM response:", response)
    return Command(update={"user_language": language, "messages": [response]})


def judge_response(state: State) -> Command:
    """Review the chatbot final text and return a clear HTML response for the user."""
    logger.debug("Entering judge node.")
    logger.info("Judge State: %s", state)


    last_ai = next((m for m in reversed(state["messages"]) if getattr(m, "type", "") == "ai"), None)
    if last_ai is None:
        logger.warning("Judge node skipped: no AI response found.")
        return Command(update={})

    candidate_response = _extract_text(getattr(last_ai, "content", ""))
    if not candidate_response.strip():
        logger.warning("Judge node skipped: chatbot response was empty.")
        return Command(update={})

    start_time = time.perf_counter()
    reviewed = cast(AIMessage, _llm.invoke([JUDGE_SYSTEM_MESSAGE, HumanMessage(content=candidate_response)]))
    elapsed = time.perf_counter() - start_time
    logger.info("Judge response completed in %.3f seconds.", elapsed)


    #print("candidate response:", candidate_response)
    #print("JUDGE response:", _extract_text(reviewed.content))

    return Command(
        update={
            "messages": [reviewed],
            "judge": {
                "original_response": candidate_response,
                "judged_response": _extract_text(reviewed.content),
            },
        }
    )



