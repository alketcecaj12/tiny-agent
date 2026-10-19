"""`tinyagent` - an interactive coding agent in your terminal (Chapter 10)."""

import argparse
import re
import shutil

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.display import NAME, Display, RichDisplay, label
from tinyagent.llm import LLM
from tinyagent.memory import MultimodalMemory
from tinyagent.planning import NativeReAct
from tinyagent.toolbox import create_native_code_tools

# --- Detect rich; fall back to a plain console if missing ---
try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Table = None

    class _PlainConsole:
        """Minimal rich.Console stand-in that strips markup."""

        _MARKUP = re.compile(r"\[/?[^\]]*\]")

        def print(self, *args, end="\n"):
            for arg in args:
                if isinstance(arg, str):
                    print(self._MARKUP.sub("", arg), end=end)

        @property
        def width(self):
            return shutil.get_terminal_size().columns

    console = _PlainConsole()


COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear conversation memory (keeps system prompt)",
    "/memory": "Show current conversation memory",
    "/trajectory": "Show what the agent did so far",
    "/model": "Show current model name",
    "/tools": "List available tools",
    "/skills": "List available skills",
}


def handle_command(cmd: str, agent: TinyAgent) -> None:
    """Handle a slash command."""
    cmd = cmd.strip().lower()

    if cmd == "/help":
        for name, desc in COMMANDS.items():
            console.print(f"  [bold]{name:<12}[/]  [dim]{desc}[/]")

    elif cmd == "/clear":
        agent.memory.messages = agent.memory.messages[:1]
        console.print("  [dim]Memory cleared (system prompt kept).[/]")

    elif cmd == "/memory":
        messages = agent.memory.get_messages()
        console.print(f"  [dim]{len(messages)} message(s) in memory:[/]")
        for i, msg in enumerate(messages):
            content = msg["content"] if isinstance(msg["content"], str) else "<multimodal>"
            console.print(f"     {i}: {msg['role']}: {content}", end="\n")

    elif cmd == "/trajectory":
        print(agent.trajectory.pretty() or "  (empty)")

    elif cmd == "/model":
        console.print(f"  [dim]{agent.llm.model} @ {agent.llm.base_url}[/]")

    elif cmd == "/tools":
        for name, tool in agent.tools.registry.items():
            if name not in getattr(agent.tools, "skills", {}):
                console.print(f"  [yellow]{name}[/] - [dim]{tool['description']}[/]")

    elif cmd == "/skills":
        skills = getattr(agent.tools, "skills", {})
        if not skills:
            console.print("  [dim]No skills loaded (use --skill PATH).[/]")
        for name, skill in skills.items():
            console.print(f"  [magenta]{name}[/] - [dim]{skill['description']}[/]")

    else:
        console.print(f"  [red]Unknown command: {cmd}[/] (type [bold]/help[/] for commands)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyagent",
        description="TinyAgent: a coding agent built from scratch.",
    )
    parser.add_argument("-m", "--model", help="LLM model name (env: TINYAGENT_MODEL)")
    parser.add_argument("-ab", "--api_base", "--base-url", dest="base_url",
                        help="OpenAI-compatible base URL (env: TINYAGENT_BASE_URL)")
    parser.add_argument("-ak", "--api_key", "--api-key", dest="api_key",
                        help="API key (env: TINYAGENT_API_KEY)")
    parser.add_argument("-w", "--workspace", default=".",
                        help="Folder the agent may read/write (default: current folder)")
    parser.add_argument("-s", "--skill", action="append", default=[],
                        help="Path to a skill folder containing SKILL.md (repeatable)")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per task")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable native reasoning (reasoning_effort=none)")
    parser.add_argument("--yes", action="store_true",
                        help="Do not ask before write_file/execute_python")
    parser.add_argument("--plain", action="store_true", help="Plain output, even if rich is installed")
    parser.add_argument("-q", "--query", help="Run a single task and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Display: rich-based if available, otherwise pure-Python ANSI
    use_rich = HAS_RICH and not args.plain
    display = RichDisplay() if use_rich else Display()

    # LLM
    think = config.env_bool("THINK", True) and not args.no_think
    llm = LLM(model=args.model, base_url=args.base_url, api_key=args.api_key, think=think)

    # Tools and Skills
    tools = create_native_code_tools(args.workspace, require_approval=not args.yes)
    for skill_path in args.skill:
        tools.add_skill(skill_path)

    # Coding Agent with native reasoning and tool calling
    agent = TinyAgent(
        llm=llm,
        memory=MultimodalMemory(),
        tools=tools,
        planner=NativeReAct(max_steps=args.max_steps),
        display=display,
    )

    def run(query: str) -> None:
        try:
            agent.run(query)
        except Exception as e:
            if hasattr(display, "stop_thinking"):
                display.stop_thinking()
            if use_rich:
                console.print(f"  {label('ERROR', 'red')}[red]{e}[/]\n")
            else:
                print(f"  ERROR    {e}\n")

    if args.query:
        run(args.query)
        return

    # Welcome message with model and tools
    print(NAME)
    console.print(f"  [dim]Model:[/]  {llm.model} @ {llm.base_url}")
    console.print(f"  [dim]Tools:[/]  {', '.join(agent.tools.registry.keys())}")
    console.print("  [dim]Type[/] /help [dim]for commands,[/] exit [dim]to quit.[/]\n")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Goodbye![/]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            console.print("  [dim]Goodbye![/]")
            break
        if query.startswith("/"):
            handle_command(query, agent)
            print()
            continue

        run(query)


if __name__ == "__main__":
    main()
