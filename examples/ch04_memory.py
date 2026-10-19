"""Chapter 4 - Memory.

Without memory the agent forgets who you are between runs; with `Memory` it
remembers the conversation.
"""

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.memory import Memory


class ForgetfulAgent:
    """The Chapter 2 agent: every run starts from a blank slate."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, task: str) -> str:
        return self.llm.generate([{"role": "user", "content": task}]).content


def main():
    # Gemma 3 12B (no native thinking or tool calling)
    llm = LLM(model=config.model("prompt"))

    intro = "Hi! We are Maarten and Jay, authors of 'An Illustrated Guide to AI Agents'."
    question = "Hi! What are our names?"

    print("== Without memory")
    agent = ForgetfulAgent(llm)
    print(agent.run(intro), "\n")
    print(agent.run(question), "\n")

    print("== With memory")
    agent = TinyAgent(llm=llm, memory=Memory())
    print(agent.run(intro), "\n")
    print(agent.run(question), "\n")

    print("== Memory contents")
    for message in agent.memory.get_messages():
        print(f"{message['role']}: {message['content']}")


if __name__ == "__main__":
    main()
