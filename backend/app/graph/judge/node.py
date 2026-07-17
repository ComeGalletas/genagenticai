from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from ..core.state import State
from ..judge.system_prompt import judge_prompt


def judge_node(state: State) -> dict:
    llm = ChatOllama(
        model="qwen3",
        temperature=0.1,
    )
    
    results_str = "\n\n".join(
        f"Title: {r.get('title')}\nContent: {r.get('content')[:500]}..."
        for r in state.get("search_results", [])[:5]
    )
    
    prompt = judge_prompt.format(query=state["messages"], results=results_str)
    
    response = llm.invoke(prompt)
    
    # Parse JSON
    import json
    try:
        decision = json.loads(str(response.content))
    except:
        decision = {"decision": "REJECT", "reason": "Failed to parse judge output"}
    
    return {
        "messages": [AIMessage(content=f"Judge: {decision['decision']} - {decision.get('reason')}")],
        "iterations": state.get("iterations", 0) + 1
    }