import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)

from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from context import AppContext, load_context
from middleware import (TodoMiddleware,
                        UsageTrackMiddleware,
                        PermissionMiddleware,
                        ContextInjectMiddleware,
                        LoggerMiddleware,
                        SkillMiddleware,
                        ContextCompactorMiddleware)
from tools import run_write, run_read, run_edit, run_glob, run_bash, run_subagent

try:
    import readline

    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass


def build_agent(context: AppContext) -> CompiledStateGraph:
    model_id = os.getenv("MODEL_ID", "deepseek-v4-flash")
    system_prompt = ("You are a coding agent. "
                     "Use tools to solve tasks. "
                     "Use subagent for focused exploration or a self-contained subtask.")
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
        tools=[run_bash, run_read, run_write, run_edit, run_glob, run_subagent],
        system_prompt=system_prompt,
        middleware=[
            ContextInjectMiddleware(),
            TodoMiddleware(),
            PermissionMiddleware(),
            LoggerMiddleware(),
            UsageTrackMiddleware(),
            SkillMiddleware(context),
            ContextCompactorMiddleware(model),
        ],
        context_schema=AppContext,
        checkpointer=InMemorySaver(),
    )


# -- Entry point --
if __name__ == "__main__":
    # print("s01: Agent Loop")
    # print("s02: Tool Use - four tools added to s01")
    # print("s04: Hooks - extension logic on hooks, loop stays clean")
    # print("s05: TodoWrite - plan before execution")
    # print("s06: Subagent - fresh messages, final text returns")
    # print("s07: Skill Loading - catalog first, full content on demand")
    # print("s08: Context Compact - archive, reduce, then summarize")

    print("Enter a question, press Enter to send. Type q to quit.\n")

    app_context: AppContext = load_context(Path.cwd())
    agent: CompiledStateGraph = build_agent(app_context)
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        try:
            query = input()
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit", ""):
            state = agent.get_state(config).values
            print(f"\033[90m[HOOK] total_tokens: {state.get('total_tokens', 0)}\033[0m")
            print(f"\033[90m[HOOK] total_input: {state.get('total_input', 0)}\033[0m")
            print(f"\033[90m[HOOK] total_output: {state.get('total_output', 0)}\033[0m")
            break

        resp = agent.invoke({"messages": query}, config=config, context=app_context)
        print(resp["messages"][-1].content)
