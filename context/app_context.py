from pydantic import BaseModel, ConfigDict

from .custom_context import (WorkspaceContext,
                             BlockCommandContext,)


class AppContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace: WorkspaceContext
    block_command: BlockCommandContext