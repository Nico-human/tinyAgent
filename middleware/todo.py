import ast
import json
from typing import TypedDict, Annotated, Literal, NotRequired, Any, Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import Field

TODO_SYSTEM_PROMPT = (
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Each call replaces the entire list, so include all tasks you want to keep. "
    "Update status as you go, using at most one in_progress task and at most 20 tasks. "
    "Call todo_write at most once per model turn. "
    "After your last tool call, give the user a final answer."
)


class TodoItem(TypedDict):
    content: Annotated[str, Field(min_length=1)]
    status: Literal["pending", "in_progress", "completed"]


class TodoState(AgentState):
    # Kept in the input schema because the CLI explicitly passes state between turns.
    todos: NotRequired[list[TodoItem]]
    rounds_since_todo: NotRequired[int]


class TodoMiddleware(AgentMiddleware[TodoState]):
    state_schema = TodoState

    def __init__(self, system_prompt: str = TODO_SYSTEM_PROMPT) -> None:
        super().__init__()
        self.tools = [TodoMiddleware.__run_todo_write]
        self.system_prompt = system_prompt

    @staticmethod
    def __render_todos(items: list[TodoItem]) -> str:
        if not items:
            return "No todos."

        lines = []
        for todo in items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[todo["status"]]
            lines.append(f"{marker} {todo['content']}")

        done = sum(todo["status"] == "completed" for todo in items)
        lines.append(f"\n({done}/{len(items)} completed)")
        return "\n".join(lines)

    @staticmethod
    def __validate_todos(todos: list | str) -> list[TodoItem]:
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as e:
                    raise ValueError("todos must be a list or JSON array string") from e

        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be an object")

            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        return validated

    @staticmethod
    @tool("todo_write")
    def __run_todo_write(
            todos: Annotated[list[TodoItem], Field(max_length=20)] | str,
            runtime: ToolRuntime[None, TodoState],
    ) -> Command:
        """Replace the full task list; use pending, in_progress, or completed statuses."""
        try:
            validated = TodoMiddleware.__validate_todos(todos)
        except ValueError as e:
            # A failed update leaves the previous todo_list intact.
            return Command(update={"messages": [ToolMessage(
                content=f"Error: {e}", tool_call_id=runtime.tool_call_id, status="error"
            )]})
        output = TodoMiddleware.__render_todos(validated)
        print(f"\n\033[33m## Current Tasks\033[0m\n{output}")
        return Command(update={
            "todos": validated,
            "messages": [ToolMessage(content=output, tool_call_id=runtime.tool_call_id)],
        })

    def before_agent(self, state: TodoState, runtime: Runtime) -> dict[str, Any]:
        # Reset the reminder counter, but preserve the session's todos list.
        return {"rounds_since_todo": 0, "todos": state.get("todos", [])}

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # inject todo_write prompt
        blocks = (
            list(request.system_message.content_blocks)
            if request.system_message is not None else []
        )
        blocks.append({"type": "text", "text": self.system_prompt})
        return handler(request.override(system_message=SystemMessage(content=blocks)))

    def after_model(self, state: TodoState, runtime: Runtime) -> dict[str, Any] | None:
        # error if call todo_write > 1 per model turn
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        calls = [c for c in last_ai.tool_calls if c["name"] == "todo_write"] if last_ai else []
        if len(calls) > 1:
            return {"messages": [ToolMessage(
                content="Error: Call todo_write only once per model turn.",
                tool_call_id=call["id"], status="error",
            ) for call in calls]}
        return None

    def before_model(self, state: TodoState, runtime: Runtime) -> dict[str, Any] | None:
        # update reminder counter, if counter >= 3, inject update message
        messages = state["messages"]
        if not messages or not isinstance(messages[-1], ToolMessage):
            return None
        last_ai = next(m for m in reversed(messages) if isinstance(m, AIMessage))
        used_todo = any(call["name"] == "todo_write" for call in last_ai.tool_calls)
        rounds = 0 if used_todo else state.get("rounds_since_todo", 0) + 1
        if rounds >= 3:
            return {
                "rounds_since_todo": 0,
                "messages": [HumanMessage("<reminder>Update your todos.</reminder>")],
            }
        return {"rounds_since_todo": rounds}
