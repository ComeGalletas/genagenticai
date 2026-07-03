from __future__ import annotations

from typing import Annotated

from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


class State(TypedDict):
    # add_messages appends new messages instead of overwriting the list,
    # which is what gives the agent its conversation memory.
    messages: Annotated[list, add_messages]
    # Populated by static_google_search each time it runs.
    searches: NotRequired[dict[str, str]]
