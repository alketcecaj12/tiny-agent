"""Terminal output for the agent's THOUGHT/ACTION/OBSERVATION/ANSWER (Chapter 10)."""

import random
import threading
import time

from tinyagent.llm import Response


# --- ANSI escape codes (used by the pure-Python Display) ---
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"  # THOUGHT
RED = "\033[31m"  # ACTION
YELLOW = "\033[33m"  # OBSERVATION
PURPLE = "\033[35m"  # ANSWER


# --- Optional rich imports (used by RichDisplay) ---
try:
    from rich.console import Console
    from rich.live import Live
    from rich.rule import Rule
    from rich.text import Text

    console = Console()
except ImportError:  # rich is optional
    Console = Live = Text = Rule = console = None


NAME = """
██████ ██ ███  ██ ██  ██   ▄████▄  ▄████  ██████ ███  ██ ██████
  ██   ██ ██ ▀▄██  ▀██▀    ██▄▄██ ██  ▄▄▄ ██▄▄   ██ ▀▄██   ██
  ██   ██ ██   ██   ██     ██  ██  ▀███▀  ██▄▄▄▄ ██   ██   ██
"""


THINKING_FRAMES = [
    "        ",
    " ~      ",
    " ~~     ",
    "~~~    ",
    "~~ ~~  ",
    "~  ~~ ~",
    "   ~~~ ",
    "     ~~",
    "        ",
]

THINKING_MESSAGES = [
    "Sonar pinging...",
    "Echolocating...",
    "Diving deep...",
    "Surfacing...",
    "Riding the current...",
    "Making a splash...",
    "Chasing fish...",
    "Blowing bubbles...",
    "Fishing for answers...",
    "Navigating the depths...",
    "Making waves...",
]


def label(name: str, style: str) -> str:
    """Format a fixed-width label for step output. (Rich markup)"""
    return f"[bold {style}]{name:<13}[/]"


class SilentDisplay:
    """Display that prints nothing (the default for library use)."""

    def __call__(self, event: str, data: str | Response | None = None) -> None:
        pass


class Display:
    """Handles agent events with ANSI-styled terminal output."""

    def __call__(self, event: str, data: str | Response | None = None) -> None:

        # "Thinking" line
        if event == "thinking":
            print(f"  {DIM}Thinking...{RESET}\n")

        # THOUGHT
        elif event == "response":
            if data.reasoning:
                print(f"{BOLD}{GREEN}{'▒▒ THOUGHT ▒▒':<13}{RESET}")
                print(f"{data.reasoning}{RESET}\n")

            # ANSWER
            if data.content:
                print(f"{BOLD}{PURPLE}{'▒▒ ANSWER ▒▒':<13}{RESET}")
                print(f"{data.content}{RESET}\n")

        # ACTION
        elif event == "tool_call" and data:
            tool = data.tool_call["tool"]
            kwargs = data.tool_call.get("kwargs")
            print(f"{BOLD}{RED}{'▒▒ ACTION ▒▒':<13}{RESET}")
            print(f"{tool}({kwargs}){RESET}\n")

        # OBSERVATION
        elif event == "observation":
            print(f"{BOLD}{YELLOW}{'▒▒ OBSERVATION ▒▒':<13}{RESET}")
            print(f"{data}{RESET}\n")
            print(f"{DIM}{'─' * 40}STEP{'─' * 40}{RESET}\n")


class RichDisplay:
    """Handles agent events with Rich formatting. Requires `pip install rich`."""

    def __init__(self):
        if Console is None:
            raise ImportError(
                "RichDisplay requires `rich`. Install with: pip install 'tinyagent[rich]'"
            )
        self._live = None
        self._animating = False
        self._thinking_msg = ""

    def __call__(self, event: str, data: str | Response | None = None) -> None:

        # Animate "thinking"
        if event == "thinking":
            self._animating = True
            self._live = Live(console=console, refresh_per_second=8, transient=True)
            self._live.start()
            self._thinking_msg = random.choice(THINKING_MESSAGES)
            threading.Thread(target=self._animate_thinking, daemon=True).start()

        # Print THOUGHT (handles both text responses and native reasoning)
        elif event == "response":
            self.stop_thinking()
            if data.reasoning:
                console.print(
                    f"  [bold dark_orange]{'THOUGHT':<13}[/][dim italic]{data.reasoning}[/]"
                )
            if data.content:
                console.print(f"\n  :dolphin: [bold cyan]ANSWER[/]    {data.content}\n")

        # Print ACTION
        elif event == "tool_call" and data:
            tool = data.tool_call["tool"]
            kwargs = data.tool_call.get("kwargs")
            console.print(f"  {label('ACTION', 'yellow')}[yellow]{tool}({kwargs})[/]")

        # Print OBSERVATION
        elif event == "observation":
            console.print(f"  {label('OBSERVATION', 'green')}{data}\n")
            console.print(Rule(style="dim"), end="\n\n")

    def _animate_thinking(self) -> None:
        """A wave animation."""
        i = 0
        while self._animating and self._live:
            frame = THINKING_FRAMES[i % len(THINKING_FRAMES)]
            live = self._live
            if live:
                live.update(Text(f"  {self._thinking_msg}  {frame}", style="dark_orange"))
            time.sleep(0.15)
            i += 1

    def stop_thinking(self) -> None:
        """Stop the thinking animation."""
        self._animating = False
        if self._live:
            self._live.stop()
            self._live = None
