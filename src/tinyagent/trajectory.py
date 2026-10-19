"""Record what the agent did, step by step (Chapter 2)."""

from dataclasses import dataclass

from tinyagent.llm import Response


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """Records agent execution as a sequence of runs.

    Each run is a dict with keys "query" (str) and "steps" (list[Step]).
    """

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation."""
        # Add THOUGHT
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)

    @property
    def steps(self) -> list[Step]:
        """All steps across all runs (flat list)."""
        return [step for run in self.runs for step in run["steps"]]

    def pretty(self) -> str:
        """A plain-text rendering of the trajectory (replaces the notebook viewer)."""
        lines = []
        for run_index, run in enumerate(self.runs, start=1):
            lines.append(f"=== Run {run_index}: {run['query']}")
            for step_index, step in enumerate(run["steps"], start=1):
                lines.append(f"--- Step {step_index}")
                if step.thought:
                    lines.append(f"  THOUGHT:     {step.thought}")
                if step.action:
                    lines.append(f"  ACTION:      {step.action}")
                    lines.append(f"  OBSERVATION: {step.observation}")
                if step.answer is not None:
                    lines.append(f"  ANSWER:      {step.answer}")
        return "\n".join(lines)
