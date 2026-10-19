"""Chapter 8 - Multi-agent collaboration.

An orchestrator agent delegates to a math specialist and a date specialist,
each exposed to it as a tool.
"""

from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct
from tinyagent.toolbox import add, days_between, multiply, subtract, today
from tinyagent.tools import NativeTools


def print_steps(name: str, agent: TinyAgent) -> None:
    print(f"== {name}")
    for run in agent.trajectory.runs:
        for index, step in enumerate(run["steps"], start=1):
            print(f"-- Step {index} ---")
            if step.action:
                print(f"Tool: {step.action}")
                print(f"Observation: {step.observation}\n")
            else:
                print(f"Answer: {step.answer}\n")


def main():
    llm = LLM(think=True)

    # Math specialist
    math_tools = NativeTools()
    math_tools.add_tool("add", add)
    math_tools.add_tool("subtract", subtract)
    math_tools.add_tool("multiply", multiply)
    math_agent = TinyAgent(llm=llm, tools=math_tools, memory=Memory(), planner=NativeReAct())

    # Date specialist
    date_tools = NativeTools()
    date_tools.add_tool("today", today)
    date_tools.add_tool("days_between", days_between)
    date_agent = TinyAgent(llm=llm, tools=date_tools, memory=Memory(), planner=NativeReAct())

    # Sub-agents as tools
    def ask_math_agent(question: str) -> str:
        """Delegate to the math specialist."""
        return math_agent.run(question)

    def ask_date_agent(question: str) -> str:
        """Ask queries to a sub-Agent that handles ISO dates."""
        return date_agent.run(question)

    tools = NativeTools()
    tools.add_tool("ask_math_agent", ask_math_agent)
    tools.add_tool("ask_date_agent", ask_date_agent)
    orchestrator = TinyAgent(llm=llm, tools=tools, memory=Memory(), planner=NativeReAct())

    print("Answer:", orchestrator.run("If I save €4 per day until 2030, how much will I have?"), "\n")
    print_steps("Orchestrator", orchestrator)
    print_steps("Date agent", date_agent)
    print_steps("Math agent", math_agent)


if __name__ == "__main__":
    main()
