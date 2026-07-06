
from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from langgraph.graph.message import add_messages

from ..retrieval.retrieval import RetrievalStage, RetrievalResult


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # Main message to be answered by the agent. This is the user query that will be used for retrieval.
    retrieval_query: NotRequired[str]
    # The result of the retrieval pipeline. This is a list of relevant passages from the knowledge base or web search.
    retrieval_results: NotRequired[list[RetrievalResult]]
    pending_pipeline: NotRequired[list[RetrievalStage]]
    completed_pipeline: NotRequired[list[RetrievalStage]]
    
    # The tool_call_id is used to track the tool call that initiated the retrieval process. It is used to associate the retrieval results with the original user query.
    tool_call_id: NotRequired[str]

