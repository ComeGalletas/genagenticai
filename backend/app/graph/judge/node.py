from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..core.state import State
from ..judge.system_prompt import JUDGE_SYSTEM_PROMPT


def judge_node(state: State) -> dict:
    """Standalone judge node kept for compatibility; core graph uses core.nodes.judge_response."""
    llm = ChatOllama(
        model="qwen3.6",
        temperature=0.1,
    )

    last_ai = next((m for m in reversed(state.get("messages", [])) if getattr(m, "type", "") == "ai"), None)
    if last_ai is None:
        return {}

    response = llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=str(getattr(last_ai, "content", ""))),
        ]
    )

    return {
        "messages": [AIMessage(content=str(response.content))],
    }