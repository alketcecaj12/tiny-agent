"""Chapter 5 - Skills.

A skill is a SKILL.md recipe the agent activates like a tool. Only its name and
description are visible up front; the instructions arrive on activation.
"""

from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM, Response
from tinyagent.memory import Memory
from tinyagent.toolbox import read_markdown
from tinyagent.tools import BUILTIN_SKILLS_DIR, Skills

URL = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/refs/heads/main/README.md"


def main():
    skill_path = BUILTIN_SKILLS_DIR / "file_analyzer"

    # Load the skill and activate it by hand (no LLM needed)
    tools_and_skills = Skills()
    tools_and_skills.add_skill(skill_path)
    print("Description:", tools_and_skills.registry["file_analyzer"]["description"])
    response = Response(content="", tool_call={"function": {"name": "file_analyzer", "arguments": {}}})
    print(tools_and_skills.execute(tools_and_skills.parse(response)), "\n")

    # Agent with a skill and the tool it needs
    tools_and_skills.add_tool("read_markdown", read_markdown)
    agent = TinyAgent(llm=LLM(), tools=tools_and_skills, memory=Memory())
    print(agent.run(f"Use the `file_analyzer` skill to summarize {URL}."))


if __name__ == "__main__":
    main()
