from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .custom_context import (
    BlockCommandContext,
    WorkspaceContext,
)
from .skill_context import Skill, load_workspace_skills


class AppContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace: WorkspaceContext
    block_command: BlockCommandContext
    skills: dict[str, Skill] = Field(default_factory=dict)


def load_context(root: Path) -> AppContext:
    workspace = WorkspaceContext(root)
    context = AppContext(
        workspace=workspace,
        block_command=BlockCommandContext(),
        skills=load_workspace_skills(workspace),
    )
    return context
