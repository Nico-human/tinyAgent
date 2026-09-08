from pathlib import Path
from typing import Callable, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from context import AppContext


class PermissionMiddleware(AgentMiddleware[AgentState, AppContext, Any]):

    def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        app_context: AppContext = request.runtime.context

        if not tool_args:
            return handler(request)

        if tool_name == "bash":
            command = tool_args.get("command", "")
            if not isinstance(command, str):
                return ToolMessage(content="Invalid argument: command must be a string",
                                   tool_call_id=request.tool_call["id"],
                                   status="error")

            for pattern in app_context.block_command.deny_list:
                if pattern in command:
                    print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                    return ToolMessage(content="Permission denied by deny list",
                                       tool_call_id=request.tool_call["id"],
                                       status="error")
            if bool(app_context.block_command.destructive_command_word.search(command)) or \
                    any(kw in command for kw in app_context.block_command.destructive):
                print(f"\n\033[33m[permission] Potentially destructive command\033[0m")
                print(f"   Tool: {tool_name}({tool_args})")
                if input("   Allow? [y/N] ").strip().lower() not in ("y", "yes"):
                    return ToolMessage(content="Permission denied by user",
                                       tool_call_id=request.tool_call["id"],
                                       status="error")
        elif tool_name in ("read_file", "edit_file", "write_file"):
            workdir: Path = app_context.workspace.root
            path = tool_args.get("path", "")
            if not isinstance(path, str):
                return ToolMessage(content="Invalid argument: path must be a string",
                                   tool_call_id=request.tool_call["id"],
                                   status="error")

            if not (workdir / path).resolve().is_relative_to(workdir):
                print(f"\n\033[33m[permission] Access outside workspace\033[0m")
                print(f"   Tool: {tool_name}({tool_args})")
                if input("   Allow? [y/N] ").strip().lower() not in ("y", "yes"):
                    return ToolMessage(content="Permission denied by user",
                                       tool_call_id=request.tool_call["id"],
                                       status="error")
        return handler(request)
