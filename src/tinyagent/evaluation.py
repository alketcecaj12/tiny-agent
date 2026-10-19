"""Evaluating the TinyAgent on small benchmarks (Chapter 7)."""

import re
from dataclasses import dataclass
from typing import Callable

from tinyagent.llm import LLM

# Type hint for the scorers: (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Scorer


class Evaluator:
    """Run a TinyAgent over a Benchmark and aggregate the results."""

    def __init__(self, create_agent: Callable):
        """Initialize with a function that creates a new agent instance."""
        self.create_agent = create_agent

    def run(self, benchmark: Benchmark) -> dict:
        """Run the agent on each example in the benchmark and score the results."""

        # Run each example and collect results
        results = []
        for example in benchmark.examples:
            agent = self.create_agent()
            prediction = agent.run(example["task"]) or ""
            passed = benchmark.scorer(prediction, example)
            results.append({"prediction": prediction, "passed": passed})

        # Aggregate pass rate
        if results:
            pass_rate = sum(result["passed"] for result in results) / len(results)
        else:
            pass_rate = 0.0

        # Return detailed results and overall pass rate
        return {"name": benchmark.name, "pass_rate": pass_rate, "results": results}


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """Return True if the answer letter matches the prediction, False otherwise."""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return bool(match) and match.group(1) == example["expected"]


def programmatic_scorer(prediction: str, example: dict) -> bool:
    """Check a prediction against its related check."""
    return example["check"](prediction)


def make_judge_scorer(judge: LLM) -> Scorer:
    """Create an LLM-as-a-judge scorer."""

    def judge_scorer(prediction: str, example: dict) -> float:
        prompt = f"""
Score the response from 0.0 to 1.0.

Expected: {example["expected"]}
Response: {prediction}

Reply with only a single number.
"""
        response = judge.generate([{"role": "user", "content": prompt}])
        match = re.search(r"\d+(?:\.\d+)?", response.content or "")
        return min(1.0, max(0.0, float(match.group()))) if match else 0.0

    return judge_scorer
