"""Chapter 6 - Native ReAct: native reasoning + native tool calling."""

from tinyagent.agent import TinyAgent
from tinyagent.display import Display
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct
from tinyagent.toolbox import add, multiply, subtract
from tinyagent.tools import NativeTools


def main():
    tools = NativeTools()
    tools.add_tool("add", add)
    tools.add_tool("subtract", subtract)
    tools.add_tool("multiply", multiply)

    agent = TinyAgent(
        llm=LLM(think=True),
        tools=tools,
        memory=Memory(),
        planner=NativeReAct(max_steps=10),
        display=Display(),
    )

    # Multi-step task with reasoning (expected: 41.996)
    answer = agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?")
    print("Final answer:", answer, "\n")
    print(agent.trajectory.pretty())


if __name__ == "__main__":
    main()
