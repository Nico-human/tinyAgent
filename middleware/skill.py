from pathlib import Path
from typing import Any, Callable, TypedDict

import yaml
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from context import AppContext


class Skill(TypedDict):
    name: str
    description: str
    content: str


class SkillMiddleware(AgentMiddleware[AgentState, AppContext, Any]):
    """Cache skills for serial invocations; use separate instances for concurrent sessions."""

    def __init__(self):
        super().__init__()
        self.skills: dict[str, Skill] = {}
        self.skill_prompt: str | None = None
        self.skill_path: Path | None = None
        # Wrap the bound method so self stays internal to the tool.
        self.tools = [tool(self.load_skills)]

    def load_skills(self, skill_name: str, runtime: ToolRuntime[AppContext, AgentState]) -> str | ToolMessage:
        """Load the full SKILL.md content by skill name."""
        skill = self.skills.get(skill_name)
        if skill is not None:
            return skill["content"]
        available = ", ".join(self.skills) or "none"
        return ToolMessage(
            content=f"Error: Unknown skill '{skill_name}'. Available: {available}",
            tool_call_id=runtime.tool_call_id,
            status="error",
        )

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict, str]:
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != "---":
            return {}, text

        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1)
             if line.rstrip("\r\n") == "---"),
            None,
        )
        if closing_index is None:
            return {}, text

        frontmatter = "".join(lines[1:closing_index])
        body = "".join(lines[closing_index + 1:]).strip()
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, body

    def before_agent(self, state: AgentState, runtime: Runtime[AppContext]) -> None:
        # 扫描skills目录, 注入到self中
        self.skill_path = runtime.context.workspace.root / "skills"
        self.skills = {}
        self.skill_prompt = None
        if not self.skill_path.exists():
            return None
        skills_root = self.skill_path.resolve()
        for manifest in sorted(self.skill_path.glob("*/SKILL.md")):
            if (not manifest.is_file()
                    or not manifest.resolve().is_relative_to(skills_root)):
                continue
            content = manifest.read_text(encoding=runtime.context.workspace.encoding)
            metadata, body = self.parse_frontmatter(content)
            raw_name = metadata.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest.parent.name
            raw_description = metadata.get("description")
            description = raw_description.strip() if isinstance(raw_description, str) else ""
            description = description or body.split("\n", 1)[0]
            description = " ".join(str(description).lstrip("# ").split())
            self.skills[name] = {
                "name": name,
                "description": description,
                "content": content,
            }
        if not self.skills:
            return None
        self.skill_prompt = "\n".join(
            f"- {skill['name']}: {skill['description']}"
            for skill in self.skills.values()
        )
        self.skill_prompt = (f"Skills available: \n{self.skill_prompt}\n\n "
                             f"Use load_skills to read the full instructions when a skill applies.")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[AppContext],
        handler: Callable[[ModelRequest[AppContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """ 注入skill prompt到系统提示词中 """
        if self.skill_prompt is None:
            return handler(request)
        blocks = (
            list(request.system_message.content_blocks)
            if request.system_message is not None else []
        )
        blocks.append({"type": "text", "text": self.skill_prompt})
        modify_request = request.override(system_message=SystemMessage(content=blocks))
        return handler(modify_request)
