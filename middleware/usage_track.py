from typing import NotRequired, Callable, Any

from langchain.messages import AIMessage

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelResponse,
    ModelRequest,
    ExtendedModelResponse
)
from langgraph.runtime import Runtime
from langgraph.types import Command


class UsageTrackState(AgentState):
    last_turn_input_tokens: NotRequired[int]
    last_turn_output_tokens: NotRequired[int]
    last_turn_total_tokens: NotRequired[int]
    last_turn_cache_tokens: NotRequired[int]


class UsageTrackMiddleware(AgentMiddleware[UsageTrackState]):
    state_schema = UsageTrackState

    def __init__(self):
        super().__init__()

    def before_agent(self, state: UsageTrackState, runtime: Runtime) -> dict[str, Any] | None:
        return {
            "last_turn_input_tokens": 0,
            "last_turn_output_tokens": 0,
            "last_turn_total_tokens": 0,
            "last_turn_cache_tokens": 0,
        }

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse],
                        ) -> ModelResponse | ExtendedModelResponse:
        response = handler(request)
        ai_msg = next((msg for msg in reversed(response.result) if isinstance(msg, AIMessage)), None)

        if ai_msg is None or (usage := ai_msg.usage_metadata) is None:
            return response

        state: UsageTrackState = request.state
        input_tokens = state.get("last_turn_input_tokens", 0) + usage["input_tokens"]
        output_tokens = state.get("last_turn_output_tokens", 0) + usage["output_tokens"]
        total_tokens = state.get("last_turn_total_tokens", 0) + usage["total_tokens"]
        cache_hits = state.get("last_turn_cache_tokens", 0) + usage.get("input_token_details", {}).get("cache_read", 0)

        return ExtendedModelResponse(model_response=response,
                                     command=Command(update={
                                         "last_turn_input_tokens": input_tokens,
                                         "last_turn_output_tokens": output_tokens,
                                         "last_turn_total_tokens": total_tokens,
                                         "last_turn_cache_tokens": cache_hits,
                                     }))

    def after_agent(self, state: UsageTrackState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"\033[90m[HOOK] last_turn_input_tokens: {state['last_turn_input_tokens']}\033[0m")
        print(f"\033[90m[HOOK] last_turn_output_tokens: {state['last_turn_output_tokens']}\033[0m")
        print(f"\033[90m[HOOK] last_turn_total_tokens: {state['last_turn_total_tokens']}\033[0m")
        print(f"\033[90m[HOOK] last_turn_cache_tokens: {state['last_turn_cache_tokens']}\033[0m")
