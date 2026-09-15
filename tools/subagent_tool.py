from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime

from context import AppContext
from middleware import PermissionMiddleware, LoggerMiddleware, UsageTrackMiddleware, \
    ContextInjectMiddleware
from state import SessionState
from .file_system_tool import run_read, run_write, run_edit
from .shell_tool import run_bash, run_glob


def build_subagent() -> CompiledStateGraph:
    submodel_id = "deepseek-v4-flash"

    sub_system = (
        f"You are a coding agent. "
        "Complete the given task, then return a concise final answer."
    )
    model = ChatDeepSeek(
        model=submodel_id,
        temperature=0.7,
        timeout=120,
        max_retries=6,
        max_tokens=80000,
        model_kwargs={"parallel_tool_calls": False},
    )
    subagent = create_agent(model=model,
                            tools=[run_bash, run_read, run_write, run_edit, run_glob],
                            system_prompt=sub_system,
                            middleware=[ContextInjectMiddleware(),
                                        PermissionMiddleware(),
                                        LoggerMiddleware(),
                                        UsageTrackMiddleware()],
                            context_schema=AppContext)
    return subagent


@tool("task")
def run_subagent(runtime: ToolRuntime[AppContext, AgentState], prompt: str) -> str | ToolMessage:
    """
    Run a subagent with fresh conversation context and return its final text.
    """
    print("\n\033[35m[Subagent started]\033[0m")
    session_state: SessionState = {"messages": []}
    session_state["messages"].append(HumanMessage(prompt))
    context: AppContext = runtime.context

    try:
        subagent = build_subagent()
        response = subagent.invoke(session_state, context = context)
        result: str | None = format_subagent_resp(response)
        return result if result else "Subagent stopped without a final answer."
    except Exception as e:
        return ToolMessage(content=f"Error: {e}", tool_call_id=runtime.tool_call_id, status="error")
    finally:
        print("\033[35m[Subagent stopped]\033[0m")


def format_subagent_resp(response: dict[str, Any]) -> str | None:
    return response["messages"][-1].content