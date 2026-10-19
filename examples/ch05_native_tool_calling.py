"""Chapter 5 - Native tool calling.

Tools are passed as JSON schemas generated from Python functions, and the
model returns structured `tool_calls`.
"""

import inspect
import json

from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.tools import NativeTools, tool_to_schema


def multiply(a: str, b: str) -> float:
    """Multiply two values."""
    return float(a) * float(b)


def get_weather(location: str) -> str:
    """Get the weather using a `location` parameter."""
    return f"Weather in {location}: Sunny, 72°F"


def main():
    # Function metadata is all we need to build a schema
    print("NAME:", get_weather.__name__)
    print("PARAMETERS:", inspect.signature(get_weather).parameters)
    print("DOCSTRING:", inspect.getdoc(get_weather), "\n")
    print(json.dumps(tool_to_schema(multiply), indent=2), "\n")

    # Register tools
    tools = NativeTools()
    tools.add_tool("multiply", multiply)
    tools.add_tool("get_weather", get_weather)

    # Gemma 4 E4B (with native thinking and tool calling)
    llm = LLM()
    agent = TinyAgent(llm=llm, memory=Memory(), tools=tools)
    print("Answer:", agent.run("What is 5.1 times 7.3?"), "\n")
    print("Answer:", agent.run("Hi! Tell me something about flamingos in two sentences."), "\n")
    print(agent.trajectory.pretty())


if __name__ == "__main__":
    main()
