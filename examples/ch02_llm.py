"""Chapter 2 - Large Language Models.

Give the TinyAgent a "brain": call an OpenAI-compatible endpoint, wrap the
answer in a `Response`, and record each step in a `Trajectory`.
"""

import json
import urllib.request

from tinyagent import config
from tinyagent.llm import LLM, Response
from tinyagent.trajectory import Trajectory


def raw_request() -> None:
    """Section 4: the raw HTTP request behind every LLM call."""
    data = json.dumps(
        {
            "model": config.model("native"),
            "messages": [{"role": "user", "content": "Hi! How's life?"}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url=f"{config.base_url().rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key()}",
        },
    )
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    print("Raw choice:", result["choices"][0], "\n")


class TinyAgent:
    """The Chapter 2 agent: an LLM and a trajectory, nothing more."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.memory = None  # Chapter 4: Add Memory
        self.tools = None  # Chapter 5: Add Tools
        self.planner = None  # Chapter 6: Add Planning

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single step."""
        messages = [{"role": "user", "content": task}]
        response = self.llm.generate(messages)
        self.trajectory.add(response)
        return response.content


def main():
    raw_request()

    # The LLM wrapper returns a `Response` dataclass
    llm = LLM(think=False)
    response: Response = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
    print("Response:", response, "\n")

    # The TinyAgent with a brain
    agent = TinyAgent(llm=llm)
    print("Answer:", agent.run("What is 2 + 2?"), "\n")
    print(agent.trajectory.pretty())


if __name__ == "__main__":
    main()
