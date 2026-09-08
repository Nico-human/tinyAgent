from typing import TypedDict, Annotated, Literal, NotRequired

from langchain.agents.middleware import AgentState
from pydantic import Field


# -- todo_state --
class TodoItem(TypedDict):
    content: Annotated[str, Field(min_length=1)]
    status: Literal["pending", "in_progress", "completed"]


class TodoState(AgentState):
    # Kept in the input schema because the CLI explicitly passes state between turns.
    todos: NotRequired[list[TodoItem]]
    rounds_since_todo: NotRequired[int]


# -- usage track state --
class UsageTrackState(AgentState):
    last_turn_input_tokens: NotRequired[int]
    last_turn_output_tokens: NotRequired[int]
    last_turn_total_tokens: NotRequired[int]
    last_turn_cache_tokens: NotRequired[int]
