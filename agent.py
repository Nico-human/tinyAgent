import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (wrap_tool_call,
                                         ToolCallRequest,
                                         before_agent)
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langgraph.runtime import Runtime
from langgraph.types import Command

from middleware import TodoMiddleware, TodoState, UsageTrackMiddleware

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

MODEL_ID = os.getenv("MODEL_ID", "deepseek-v4-flash")
WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


@wrap_tool_call
def permission_middleware(request: ToolCallRequest,
                          handler: Callable[[ToolCallRequest], ToolMessage | Command],
                          ) -> ToolMessage | Command:
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})

    if not tool_args:
        return handler(request)

    if tool_name == "bash":
        command = tool_args.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return ToolMessage(content = "Permission denied by deny list",
                                   tool_call_id = request.tool_call["id"])
        if bool(DESTRUCTIVE_COMMAND_WORD.search(command)) or \
           any(kw in command for kw in DESTRUCTIVE):
            print(f"\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {tool_name}({tool_args})")
            if input("   Allow? [y/N] ").strip().lower() not in ("y", "yes"):
                return ToolMessage(content = "Permission denied by user",
                                   tool_call_id = request.tool_call["id"])
    elif tool_name in ("read_file", "edit_file", "write_file"):
        path = tool_args.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {tool_name}({tool_args})")
            if input("   Allow? [y/N] ").strip().lower() not in ("y", "yes"):
                return ToolMessage(content = "Permission denied by user",
                                   tool_call_id = request.tool_call["id"])
    return handler(request)

@wrap_tool_call
def log_middleware(request: ToolCallRequest,
                   handler: Callable[[ToolCallRequest], ToolMessage | Command],
                   ) -> ToolMessage | Command:
    """PreToolUse: log every tool call."""
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    args_preview = "  ".join(f"{k}: {v}" for k, v in tool_call["args"].items())[:60]
    print(f"\033[90m[MIDDLEWARE] {tool_name}({args_preview})\033[0m")
    return handler(request)


@wrap_tool_call
def large_output_middleware(request: ToolCallRequest,
                            handler: Callable[[ToolCallRequest], ToolMessage | Command]
                            ) -> ToolMessage | Command:
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    response = handler(request)
    if isinstance(response, ToolMessage):
        content_length = len(str(response.content))
        if content_length > 100000:
            print(f"\033[33m[HOOK] Large output from {tool_name}: {content_length} chars\033[0m")
    return response


@before_agent
def context_inject_prompt(state: AgentState, runtime: Runtime) -> None:
    # TODO: 动态注入提示词
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")

# -- File and shell tools --

@tool("bash")
def run_bash(command: str) -> str:
    """
    Run a shell command.
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """
    Read file contents.
    """
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if limit and 0 <= limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool("write_file")
def run_write(path: str, content: str) -> str:
    """
    write content to a file.
    """
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("edit_file")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file once.
    """
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool("glob")
def run_glob(pattern: str) -> str:
    """
    Find files matching a glob pattern; ** matches recursively.
    """
    import glob as g
    try:
        matches = sorted({
            match for match in g.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

def build_agent(model: BaseChatModel | None = None):
    """Build an agent; accepting a model also allows offline testing."""
    if model is None:
        model = ChatDeepSeek(
            model=MODEL_ID,
            temperature=0.7,
            timeout=120,
            max_retries=6,
            max_tokens=80000,
            model_kwargs={"parallel_tool_calls": False},
        )

    return create_agent(
        model=model,
        tools=[run_bash, run_read, run_write, run_edit, run_glob],
        middleware=[
            context_inject_prompt,
            TodoMiddleware(),
            permission_middleware,
            log_middleware,
            large_output_middleware,
            UsageTrackMiddleware(),
        ],
        system_prompt=SYSTEM,
    )


# -- Entry point --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("s02: Tool Use - four tools added to s01")
    print("s04: Hooks - extension logic on hooks, loop stays clean")
    print("s05: TodoWrite - plan before execution")

    print("Enter a question, press Enter to send. Type q to quit.\n")

    model = ChatDeepSeek(
        model=MODEL_ID,
        temperature=0.7,
        timeout=120,
        max_retries=6,
        max_tokens=80000,
        model_kwargs={"parallel_tool_calls": False},
    )
    agent = build_agent(model=model)
    session_state: TodoState = {"messages": []}

    while True:
        try:
            query = input()
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit", ""):
            break

        session_state["messages"].append(HumanMessage(query))
        session_state = agent.invoke(session_state)
        print(session_state["messages"][-1].content)