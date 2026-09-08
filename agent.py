import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph

from context import AppContext, WorkspaceContext, BlockCommandContext
from middleware import TodoMiddleware, UsageTrackMiddleware, PermissionMiddleware, ContextInjectMiddleware, \
    LoggerMiddleware
from state import SessionState
from tools import run_write, run_read, run_edit, run_glob, run_bash

try:
    import readline

    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

load_dotenv(verbose=True, override=True)


def build_agent() -> CompiledStateGraph:
    model_id = os.getenv("MODEL_ID", "deepseek-v4-flash")
    system_prompt = "You are a coding agent. Use tools to solve tasks."
    model = ChatDeepSeek(
        model=model_id,
        temperature=0.7,
        timeout=120,
        max_retries=6,
        max_tokens=80000,
        model_kwargs={"parallel_tool_calls": False},
    )

    return create_agent(
        model=model,
        tools=[run_bash, run_read, run_write, run_edit, run_glob],
        system_prompt=system_prompt,
        middleware=[
            ContextInjectMiddleware(),
            TodoMiddleware(),
            PermissionMiddleware(),
            LoggerMiddleware(),
            UsageTrackMiddleware(),
        ],
        context_schema=AppContext,
    )


def get_context_info() -> AppContext:
    block_command = BlockCommandContext()
    context = AppContext(workspace=WorkspaceContext(Path.cwd()), block_command=block_command)
    return context


# -- Entry point --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("s02: Tool Use - four tools added to s01")
    print("s04: Hooks - extension logic on hooks, loop stays clean")
    print("s05: TodoWrite - plan before execution")

    print("Enter a question, press Enter to send. Type q to quit.\n")

    agent = build_agent()
    session_state: SessionState = {"messages": []}
    app_context = get_context_info()

    while True:
        try:
            query = input()
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit", ""):
            break

        session_state["messages"].append(HumanMessage(query))
        session_state = agent.invoke(input=session_state, context=app_context)
        print(session_state["messages"][-1].content)
