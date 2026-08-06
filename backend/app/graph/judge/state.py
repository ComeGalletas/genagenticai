from typing_extensions import TypedDict


class JudgeState(TypedDict):
    original_response: str
    judged_response: str
