from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt")
    thread_id: str = Field("default", description="Conversation thread ID for memory")


class ChatResponse(BaseModel):
    reply: str
