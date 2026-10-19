"""Chapter 7 - Evaluation: exact match, programmatic checks and LLM-as-a-judge.

The judge uses TINYAGENT_JUDGE_MODEL / TINYAGENT_JUDGE_BASE_URL /
TINYAGENT_JUDGE_API_KEY when set, and the main LLM otherwise.
"""

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.evaluation import (
    Benchmark,
    Evaluator,
    exact_match_scorer,
    make_judge_scorer,
    programmatic_scorer,
)
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct
from tinyagent.tools import NativeTools

llm = LLM(think=True)


def create_agent() -> TinyAgent:
    """Create a new instance of TinyAgent"""
    return TinyAgent(llm=llm, memory=Memory(), tools=NativeTools(), planner=NativeReAct())


MMLU_PRO = Benchmark(
    name="MMLU Pro",
    examples=[
        {
            "task": """Which of the following is the body cavity that contains the pituitary gland?
A) Ventral B) Dorsal C) Buccal D) Thoracic E) Pericardial F) Abdominal G) Spinal H) Pelvic I) Pleural J) Cranial
Answer with only the letter.""",
            "expected": "J",
        },
        {
            "task": """What is the approximate mean cranial capacity of Homo erectus?
A) 1200 cc B) under 650 cc C) 1700 cc D) 1350 cc E) just under 1000 cc F) 1500 cc G) under 500 cc H) about 800 cc I) just over 1100 cc J) about 900 cc
Answer with only the letter.""",
            "expected": "E",
        },
        {
            "task": """According to Moore's "ideal utilitarianism," the right action is the one that brings about the greatest amount of:
A) wealth. B) virtue. C) fairness. D) pleasure. E) peace. F) justice. G) happiness. H) power. I) good. J) knowledge.
Answer with only the letter.""",
            "expected": "I",
        },
    ],
    scorer=exact_match_scorer,
)

IFEVAL = Benchmark(
    name="IFeval",
    examples=[
        {
            "task": "Write me a funny song with less than 10 sentences for a proposal to build a new playground at my local elementary school.",
            "check": lambda text: sum(1 for c in text if c in ".!?") < 10,
        },
        {
            "task": "Write an ad copy for a new product, a digital photo frame that connects to your social media accounts and displays your photos. Respond with at most 150 words.",
            "check": lambda text: len(text.split()) <= 150,
        },
        {
            "task": "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
            "check": lambda text: "," not in text,
        },
    ],
    scorer=programmatic_scorer,
)


def judge_benchmark() -> Benchmark:
    judge = LLM(
        model=config.env("JUDGE_MODEL"),
        base_url=config.env("JUDGE_BASE_URL"),
        api_key=config.env("JUDGE_API_KEY"),
        think=None,
    )
    return Benchmark(
        name="MMLU Pro (LLM-as-a-judge)",
        examples=[
            {"task": "Which body cavity contains the pituitary gland?", "expected": "the cranial cavity"},
            {"task": "What is the approximate mean cranial capacity of Homo erectus?", "expected": "just under 1000 cc"},
            {"task": "According to Moore's 'ideal utilitarianism,' the right action is the one that brings about the greatest amount of what?", "expected": "good"},
        ],
        scorer=make_judge_scorer(judge),
    )


def report(result: dict) -> None:
    print(f"== {result['name']}: pass rate {result['pass_rate']:.2f}")
    for item in result["results"]:
        prediction = item["prediction"].replace("\n", " ")
        print(f"  [{item['passed']}] {prediction[:100]}")
    print()


def main():
    evaluator = Evaluator(create_agent)
    for benchmark in (MMLU_PRO, IFEVAL, judge_benchmark()):
        report(evaluator.run(benchmark))


if __name__ == "__main__":
    main()
