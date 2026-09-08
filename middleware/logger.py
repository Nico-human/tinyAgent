from typing import Callable, Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from context import AppContext


class LoggerMiddleware(AgentMiddleware[AgentState, AppContext, Any]):

    def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        args_preview = "  ".join(f"{k}: {v}" for k, v in tool_call["args"].items())[:60]
        print(f"\033[90m[MIDDLEWARE] {tool_name}({args_preview})\033[0m")

        response = handler(request)

        if isinstance(response, ToolMessage):
            content_length = len(str(response.content))
            if content_length > 100000:
                print(f"\033[33m[HOOK] Large output from {tool_name}: {content_length} chars\033[0m")
        return response
