"""Chapter 10 - Coding agents.

A coding agent with file and Python tools, human-in-the-loop approval for
risky tools, and a terminal display. Type `exit` to quit.

For the full-featured version run the `tinyagent` command instead.
"""

import subprocess
import sys
from pathlib import Path

from tinyagent.agent import TinyAgent
from tinyagent.display import NAME, Display
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct
from tinyagent.tools import NativeTools


def read_file(path: str) -> str:
    """Read a file's contents."""
    target = Path(path)
    if not target.exists():
        return f"Error: '{path}' not found."
    return target.read_text(encoding="utf-8")


def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    target = Path(directory)
    if not target.is_dir():
        return f"Error: '{directory}' is not a directory."
    entries = sorted(target.iterdir())
    return "\n".join(p.name + ("/" if p.is_dir() else "") for p in entries) or "(empty)"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written to '{path}'."


def execute_python(code: str) -> str:
    """Execute Python code and return output."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."
    if result.returncode != 0:
        return (
            f"Exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        ).strip()
    return result.stdout.strip() or "(no output)"


def main():
    # Some tools require approval
    tools = NativeTools(requires_approval=["write_file", "execute_python"])
    tools.add_tool("read_file", read_file)
    tools.add_tool("list_files", list_files)
    tools.add_tool("write_file", write_file)
    tools.add_tool("execute_python", execute_python)

    agent = TinyAgent(
        llm=LLM(think=True),
        tools=tools,
        memory=Memory(),
        planner=NativeReAct(),
        display=Display(),
    )

    print(f"\n{NAME}\n")
    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        try:
            agent.run(query)
        except Exception as e:
            print(f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
