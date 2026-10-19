"""Tools and Skills (Chapter 5)."""

import inspect
import json
from pathlib import Path
from typing import Any, Callable

from tinyagent.llm import Response


# Convert specific types to string descriptions
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(function: Callable, name: str | None = None) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)

    # Extract metadata
    properties, required = {}, []
    for param_name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        properties[param_name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(param_name)

    # Fill schema
    schema = {
        "type": "function",
        "function": {
            "name": name or function.__name__,
            "description": inspect.getdoc(function) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return schema


class Tools:
    """Tool registry for the Agent (prompt-based tool calling)."""

    def __init__(self, requires_approval: list[str] | set[str] | None = None):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.requires_approval = set(requires_approval or [])

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the Agent can use.

        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
        """
        self.registry[name] = {
            "function": func,
            "description": description or inspect.getdoc(func) or "",
        }

    @property
    def descriptions(self) -> str:
        """Get descriptions of all registered tools."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}" for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        return f"""
# Tools

If needed, you can only use the following tools to assist you in completing tasks:

{self.descriptions}

To use a tool, respond with JSON: {{"tool": "name", "kwargs": {{"param": "value"}}}}
"""

    def parse(self, response: Response) -> Response:
        """Parse a JSON tool call from text."""
        text = response.content or ""

        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            try:
                tool_call = json.loads(text[start:end])
            except json.JSONDecodeError:
                # Small models sometimes emit a trailing comma (as in the ReAct prompt)
                tool_call = json.loads(_strip_trailing_commas(text[start:end]))

            # Add the parsed tool call to the response
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
                metadata=response.metadata,
            )

        return response

    def execute(self, response: Response) -> Any:
        """Run a registered tool.

        Arguments:
            response: A Response whose `tool_call` has "tool" and "kwargs" keys.
        """
        tool_call = response.tool_call
        name, kwargs = tool_call["tool"], tool_call.get("kwargs") or {}

        # Human-in-the-loop: ask before running dangerous tools
        if name in self.registry and name in self.requires_approval:
            answer = input(f"Allow {name}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # Handle registered tools
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            try:
                if isinstance(kwargs, dict):
                    return tool_func(**kwargs)
                return tool_func(kwargs)
            except Exception as error:  # Let the agent see the error and recover
                return f"Error while running '{name}': {type(error).__name__}: {error}"

        return f"Tool '{name}' not found."

    def observation(self, result: Any) -> tuple[str, str]:
        """Return the observation as a user."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """The `TinyAgent` is done if there is no tool call,
        or if it uses the `final_answer` tool."""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            answer = response.tool_call.get("kwargs", "")
            # Some models wrap the answer, e.g. {"answer": "..."}
            if isinstance(answer, dict) and len(answer) == 1:
                answer = next(iter(answer.values()))
            response.content = answer if isinstance(answer, str) else json.dumps(answer)
            return True
        return False

    @property
    def schemas(self) -> list[dict] | None:
        """Used only for native tool-calling."""
        return None


def _strip_trailing_commas(text: str) -> str:
    import re

    return re.sub(r",\s*([}\]])", r"\1", text)


class NativeTools(Tools):
    """Tool registry using native function calling."""

    @property
    def schemas(self) -> list[dict]:
        """Return tool functions for native function calling."""
        return [
            tool_to_schema(tool["function"], name)
            for name, tool in self.registry.items()
        ]

    @property
    def prompt(self) -> str:
        """Empty because we don't need a prompt for native tool calling"""
        return ""

    def parse(self, response: Response) -> Response:
        """Parse a tool call."""
        # If there's no tool call, return the response as is
        if not response.tool_call:
            return response

        # Extract the tool name and arguments from the tool call
        args = response.tool_call["function"].get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        tool_call = {
            "id": response.tool_call.get("id"),
            "tool": response.tool_call["function"]["name"],
            "kwargs": args,
        }

        # Add the parsed tool call to the response
        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call=tool_call,
            metadata=response.metadata,
        )

    def observation(self, result: Any) -> tuple[str, str]:
        """Native tool results use the 'tool' role."""
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        """No tool call means the `TinyAgent` is done."""
        return not response.tool_call


def _parse_frontmatter(text: str) -> dict:
    """Parse simple `key: value` YAML frontmatter without external deps."""
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


class Skills(NativeTools):
    """A `Tools` and `Skills` registry.

    Skills are recipes. When you activate one, you get instructions
    in return on how to approach a given task. Although it is not
    a tool in the same way a calculator is one, we can still approach
    it as such since the Agent has to decide when to activate it.

    The skills are loaded progressively. As such, the name and description are
    available in the system prompt, but the full instructions are only injected
    when the agent **activates** a skill (uses it as a tool).
    """

    def __init__(self, requires_approval: list[str] | set[str] | None = None):
        super().__init__(requires_approval)
        self.skills: dict[str, dict] = {}

    def add_skill(self, path: str | Path) -> str:
        """Load a skill folder (or its SKILL.md) and register it as a tool."""
        path = Path(path)
        skill_dir = path if path.is_dir() else path.parent
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

        # TIER 2: Instructions - Full instructions of the main SKILL.md
        _, frontmatter, instructions = content.split("---", 2)
        meta = _parse_frontmatter(frontmatter)
        instructions = instructions.strip()

        # TIER 1: Catalog - Name and description of a skill
        name, description = meta["name"], meta["description"]

        # TIER 3: List files in the same folder that the agent can access
        extra_files = sorted(p.name for p in skill_dir.iterdir() if p.name != "SKILL.md")

        # When activated, return the instructions plus the folder context
        def skill(**kwargs):
            text = instructions
            if extra_files:
                text += f"\n\nSkill folder: {skill_dir}\n\nAvailable files: {extra_files}"
            return text

        skill.__name__ = name
        skill.__doc__ = "A skill. " + description
        skill.__signature__ = inspect.Signature()  # schema sees no params
        self.add_tool(name, skill, skill.__doc__)
        self.skills[name] = {"description": description, "path": str(skill_dir)}
        return name

    @property
    def prompt(self) -> str:
        if not self.skills:
            return ""
        return """You have specialized skills available.
To activate a skill, call it with an empty argument object.
Skills return instructions; they do not process the task directly."""


BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"
