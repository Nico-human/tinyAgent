from pathlib import Path

from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool

from context import AppContext

# -- file system tool --


@tool("read_file")
def run_read(
    runtime: ToolRuntime[AppContext, AgentState], path: str, limit: int | None = None
) -> str:
    """
    Read file contents.
    :arg path: file path.
    :arg limit: maximum number of lines to read. If None or ≤ 0, read all lines.
    """
    try:
        workdir: Path = runtime.context.workspace.root
        file_path = (workdir / path).resolve()
        lines = file_path.read_text(
            encoding=runtime.context.workspace.encoding
        ).splitlines()
        if limit is not None and 0 < limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(
    path: str, content: str, runtime: ToolRuntime[AppContext, AgentState]
) -> str:
    """
    write content to a file.
    """
    try:
        workdir: Path = runtime.context.workspace.root
        file_path = (workdir / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=runtime.context.workspace.encoding)
        return f"Wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(
    path: str,
    old_text: str,
    new_text: str,
    runtime: ToolRuntime[AppContext, AgentState],
) -> str:
    """
    Replace exact text in a file once.
    """
    try:
        workdir: Path = runtime.context.workspace.root
        encoding = runtime.context.workspace.encoding
        file_path = (workdir / path).resolve()
        text = file_path.read_text(encoding=encoding)
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding=encoding)
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"
