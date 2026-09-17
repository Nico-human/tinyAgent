from collections.abc import Callable
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime

from context import AppContext


class SkillMiddleware(AgentMiddleware[AgentState, AppContext, Any]):
    """Build the catalog once; invoke the agent with the same skill context."""

    def __init__(self, context: AppContext):
        super().__init__()
        self.skill_prompt: str | None = None
        if context.skills:
            catalog = "\n".join(
                f"- {skill['name']}: {skill['description']}"
                for skill in context.skills.values()
            )
            self.skill_prompt = (
                f"Skills available: \n{catalog}\n\n "
                "Use load_skills to read the full instructions when a skill applies."
            )
        self.tools = [tool(self.load_skills)]

    def load_skills(
        self, skill_name: str, runtime: ToolRuntime[AppContext, AgentState]
    ) -> str | ToolMessage:
        """Load the full SKILL.md content by skill name."""
        skills = runtime.context.skills
        skill = skills.get(skill_name, None)
        if skill is not None:
            return skill["content"]
        available = ", ".join(skills) or "none"
        return ToolMessage(
            content=f"Error: Unknown skill '{skill_name}'. Available: {available}",
            tool_call_id=runtime.tool_call_id,
            status="error",
        )

    def wrap_model_call(
        self,
        request: ModelRequest[AppContext],
        handler: Callable[[ModelRequest[AppContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """注入skill prompt到系统提示词中"""
        if self.skill_prompt is None:
            return handler(request)
        blocks = (
            list(request.system_message.content_blocks)
            if request.system_message is not None
            else []
        )
        blocks.append({"type": "text", "text": self.skill_prompt})
        modify_request = request.override(system_message=SystemMessage(content=blocks))
        return handler(modify_request)
