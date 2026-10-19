"""Chapter 6 - Planning with ReAct (prompt-based THOUGHT/ACTION/OBSERVATION)."""

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.display import Display
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import ReAct
from tinyagent.toolbox import add, multiply, subtract
from tinyagent.tools import Tools


def main():
    tools = Tools()
    tools.add_tool("add", add, "add(a: str, b: str)")
    tools.add_tool("subtract", subtract, "subtract(a: str, b: str)")
    tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

    agent = TinyAgent(
        llm=LLM(model=config.model("prompt")),
        tools=tools,
        memory=Memory(),
        planner=ReAct(max_steps=10),
        display=Display(),
    )

    # Multi-step task with reasoning (expected: 41.996)
    answer = agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?")
    print("Final answer:", answer, "\n")
    print(agent.trajectory.pretty())


if __name__ == "__main__":
    main()
