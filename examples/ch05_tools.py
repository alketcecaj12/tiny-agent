"""Chapter 5 - Tools (prompt-based tool calling).

The tools are described in the system prompt and the LLM answers with a JSON
blob such as {"tool": "multiply", "kwargs": {"a": "5.1", "b": "7.3"}}.
"""

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM, Response
from tinyagent.memory import Memory
from tinyagent.toolbox import get_weather, multiply
from tinyagent.tools import Tools


def main():
    # Register tools
    tools = Tools()
    tools.add_tool("multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)")
    tools.add_tool("get_weather", get_weather, "Gets weather: get_weather(location: str)")
    print(tools.prompt)

    # Parse and execute a fake LLM response (no LLM needed)
    response = Response(content='{"tool": "multiply", "kwargs": {"a": "3.1", "b": "6.5"} }')
    response = tools.parse(response)
    print("Fake tool call result:", tools.execute(response), "\n")

    # The agent with tools
    llm = LLM(model=config.model("prompt"))
    agent = TinyAgent(llm=llm, memory=Memory(), tools=tools)
    print("Answer:", agent.run("What is 5.1 times 7.3?"), "\n")
    print("Answer:", agent.run("Hi! Tell me something about flamingos in two sentences."), "\n")
    print(agent.trajectory.pretty())


if __name__ == "__main__":
    main()
