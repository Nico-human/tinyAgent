from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from context import AppContext


# -- inject context --

class ContextInjectMiddleware(AgentMiddleware[AgentState, AppContext, Any]):

    def before_agent(self, state: AgentState, runtime: Runtime[AppContext]) -> dict[str, Any] | None:
        # TODO: 动态注入提示词
        workdir: Path = runtime.context.workspace.root
        print(f"\033[90m[HOOK] UserPromptSubmit: working in {workdir}\033[0m")
