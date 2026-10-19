"""Chapter 4 - Short-term memory: trimming and summarization."""

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.memory import SummarizationMemory, TrimmingMemory


def show(messages: list[dict]) -> None:
    for message in messages:
        print(f"  {message['role']}: {message['content']}")
    print()


def main():
    llm = LLM(model=config.model("prompt"))

    print("== TrimmingMemory: keeps the system prompt + last two turns")
    agent = TinyAgent(llm=llm, memory=TrimmingMemory())
    agent.run("Hi! I'm reading 'An Illustrated Guide to AI Agents'.")
    agent.run("There are many flamingos in this book, why?")
    agent.run("What is 1+1?")
    agent.run("What is 2+2?")
    show(agent.memory.get_messages())

    print("== SummarizationMemory: compresses everything into a running summary")
    agent = TinyAgent(llm=llm, memory=SummarizationMemory(llm=llm))
    print(agent.run("Hi, my name is Sarah. Why are flamingos pink?"), "\n")
    show(agent.memory.get_messages())

    agent.run("Tell me about yourself.")
    agent.run("What is your favorite color?")
    agent.run("Are you fully autonomous?")
    show(agent.memory.get_messages())


if __name__ == "__main__":
    main()
