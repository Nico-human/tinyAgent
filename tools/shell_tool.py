import subprocess
from pathlib import Path

from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool

from context import AppContext

# -- shell tools --


@tool("bash")
def run_bash(command: str, runtime: ToolRuntime[AppContext, AgentState]) -> str:
    """
    Run a shell command.
    """
    dangerous = runtime.context.block_command.deny_list
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        workdir: Path = runtime.context.workspace.root
        r = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return (
            f"return code: {r.returncode}\n output: {out[:50000]}"
            if out
            else f"return code: {r.returncode}\n (no output)"
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


@tool("glob")
def run_glob(pattern: str, runtime: ToolRuntime[AppContext, AgentState]) -> str:
    """
    Find files matching a glob pattern; ** matches recursively.
    """
    import glob as g

    try:
        workdir: Path = runtime.context.workspace.root
        matches = sorted(
            {
                match
                for match in g.glob(pattern, root_dir=workdir, recursive=True)
                if (workdir / match).resolve().is_relative_to(workdir)
            }
        )
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"
