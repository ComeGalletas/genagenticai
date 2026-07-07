from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """ Creates the schema for the chat request body.
        Sends the thread_id to the backend to keep track of the conversation history. If no thread_id is provided, it defaults to "default".
    """
    message: str = Field(..., min_length=1, description="User prompt")
    thread_id: str = Field("default", description="Conversation thread ID for memory")

class ChatResponse(BaseModel):
    """Creates the schema for the chat response body."""
    reply: str
