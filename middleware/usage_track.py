from typing import Callable, Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelResponse,
    ModelRequest,
    ExtendedModelResponse
)
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from context import AppContext
from state import UsageTrackState


class UsageTrackMiddleware(AgentMiddleware[UsageTrackState, AppContext, Any]):
    state_schema = UsageTrackState

    def __init__(self):
        super().__init__()

    def before_agent(self, state: UsageTrackState, runtime: Runtime[AppContext]) -> dict[str, Any] | None:
        return {
            "last_turn_input_tokens": 0,
            "last_turn_output_tokens": 0,
            "last_turn_total_tokens": 0,
            "last_turn_cache_tokens": 0,
        }

    def wrap_model_call(self, request: ModelRequest[AppContext], handler: Callable[[ModelRequest[AppContext]], ModelResponse],
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

    def after_agent(self, state: UsageTrackState, runtime: Runtime[AppContext]) -> dict[str, Any] | None:
        print(f"\033[90m[HOOK] last_turn_input_tokens: {state['last_turn_input_tokens']}\033[0m")
        print(f"\033[90m[HOOK] last_turn_output_tokens: {state['last_turn_output_tokens']}\033[0m")
        print(f"\033[90m[HOOK] last_turn_total_tokens: {state['last_turn_total_tokens']}\033[0m")
        print(f"\033[90m[HOOK] last_turn_cache_tokens: {state['last_turn_cache_tokens']}\033[0m")
        return {
            "total_tokens": state.get("total_tokens", 0) + state["last_turn_total_tokens"],
            "total_input": state.get("total_input", 0) + state["last_turn_input_tokens"],
            "total_output": state.get("total_output", 0) + state["last_turn_output_tokens"],
        }