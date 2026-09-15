from typing import Annotated

from langgraph.graph.message import MessagesState
from typing_extensions import NotRequired

from ...config import RETRIEVAL_MAX_ENTRIES
from ..judge.state import JudgeState
from ..retrieval.state import RetrievalState


def bounded_retrievals(left: list[RetrievalState] | None, right: list[RetrievalState] | None) -> list[RetrievalState]:
    """Reducer for `retrieval`: append new tool results, keep only the newest entries, and
    reset when a node writes None.

    A bounded window instead of a per-turn reset keeps follow-up questions ("and its TDP?")
    answerable from the previous search while stopping the context from growing forever.
    """
    if right is None:
        return []
    merged = list(left or []) + list(right)
    if RETRIEVAL_MAX_ENTRIES > 0:
        return merged[-RETRIEVAL_MAX_ENTRIES:]
    return merged


class State(MessagesState):
    user_language: NotRequired[str | None]
    tool_call_id: NotRequired[str]
    # Retrieval tool results that the chatbot and judge see as context. Bounded, see reducer.
    retrieval: NotRequired[Annotated[list[RetrievalState], bounded_retrievals]]
    judge: NotRequired[JudgeState | None]
    # How many times this turn's answer was sent back to the chatbot after a failed verdict.
    judge_attempts: NotRequired[int]
    # Sanitized HTML produced by the finalize node; what the API returns to the client.
    final_response: NotRequired[str | None]
