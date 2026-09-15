import logging
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage

from ...config import JUDGE_MAX_RETRIES, JUDGE_SKIP_MAX_CHARS
from ...config import USE_RETRIEVAL_PIPELINE_TOOL  # to be deprecated: switches the retrieval pipeline between tool and node mode.
from ...retrieval.service import retrieval_engine
from ..judge.critique import is_critique
from .state import State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def turn_messages(state: State) -> list[BaseMessage]:
    """Messages produced since the user's latest real message (judge critiques do not start a turn)."""
    messages = state["messages"]
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if getattr(message, "type", "") == "human" and not is_critique(message):
            return messages[index + 1:]
    return list(messages)


def turn_tool_call_ids(state: State) -> set[str]:
    """Ids of every tool call the assistant made since the user's latest real message."""
    ids: set[str] = set()
    for message in turn_messages(state):
        if getattr(message, "type", "") == "ai":
            ids.update(call["id"] for call in (getattr(message, "tool_calls", None) or []) if call.get("id"))
    return ids


def turn_retrievals(state: State) -> list[dict]:
    """Retrieval entries produced by this turn's tool calls (the window may also hold earlier turns')."""
    ids = turn_tool_call_ids(state)
    return [entry for entry in (state.get("retrieval") or []) if entry.get("tool_call_id") in ids]


def web_searched_this_turn(state: State) -> bool:
    """True when a retrieval this turn already reached the last (web) stage, so escalation is impossible."""
    last_stage = len(retrieval_engine.pipeline) - 1
    return any(entry.get("retrieval_stage") == last_stage for entry in turn_retrievals(state))


def should_judge(state: State, answer: AIMessage) -> bool:
    """Deterministic gate: judge tool-backed answers and long answers; skip short small talk."""
    used_tools = any(getattr(m, "type", "") == "tool" for m in turn_messages(state))
    length = len(_text(answer.content).strip())
    if used_tools:
        return True
    if length >= JUDGE_SKIP_MAX_CHARS:
        return True
    logger.info("Judge skipped | no tools used and reply is %d chars (< %d)", length, JUDGE_SKIP_MAX_CHARS)
    return False


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def route_chatbot(state: State) -> str:
    """After the chatbot: run requested tools, judge a final answer, or finalize small talk directly."""
    last = cast(AIMessage, state["messages"][-1])

    if getattr(last, "tool_calls", None):
        tool_name = last.tool_calls[0]["name"]
        if not USE_RETRIEVAL_PIPELINE_TOOL and tool_name == "retrieve_information":
            return "retrieve_information"
        return "tools"

    return "judge" if should_judge(state, last) else "finalize"


def route_judge(state: State) -> str:
    """After the judge: finalize a passing answer, or send a failed one back once for revision."""
    verdict = state.get("judge")
    if verdict is None or verdict.get("passed", True):
        return "finalize"

    attempts = state.get("judge_attempts", 0)
    if attempts >= JUDGE_MAX_RETRIES:
        logger.info("Judge failed again after %d revision(s); returning the answer anyway.", attempts)
        return "finalize"

    return "revise"


def retrieval_router(state: State) -> str:
    """After the retrieve_information node (node mode only): keep escalating stages or finish."""
    # A None stage means the pipeline has run out of stages.
    if state.get("retrieval_stage") is None:
        return "finish_retrieval"

    if retrieval_engine._enough_information(state.get("retrieval_results", [])):
        return "finish_retrieval"

    return "retrieve_information"
