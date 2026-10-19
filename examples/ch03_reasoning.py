"""Chapter 3 - Reasoning LLMs.

Few-shot prompting, (zero-shot) chain-of-thought, self-consistency, best-of-N
and native reasoning.

    python examples/ch03_reasoning.py            # everything, 10 samples
    python examples/ch03_reasoning.py --samples 3
"""

import argparse
from collections import Counter

from tinyagent import config
from tinyagent.llm import LLM

PENGUINS = (
    "I saw 9 penquins. 2 slid in the water and disappeared from sight whilst "
    "4 waddled up from the shore. How many can I see?"
)

SEATING = """Six friends (Maarten, Ilse, Sarah, Jor, Irene, and Chris) are sitting in a row.

- Sarah is in seat 3.
- Chris is sitting at one of the ends of the row.
- Jor is sitting immediately to the right of Chris.
- Ilse is sitting exactly in the middle of Sarah and Maarten.
- Irene is not sitting next to Sarah.
- Maarten is sitting somewhere to the left of Irene.

In which seat is Maarten sitting?
Let's think step by step and give back your answer only after 'Answer:'.
"""

ROMAN = """Write a Python function `roman_to_int(s)` that converts a Roman numeral
string to an integer. The function should:
- Handle standard Roman numerals from 1 to 3999.
- Correctly handle subtractive notation (e.g., IV=4, IX=9, XL=40, XC=90, CD=400, CM=900).
- Return 0 for invalid inputs (empty string, non-Roman characters, or invalid patterns
  like "IIII" or "VV").

Give only the function definition, no explanation or markdown formatting."""

TEST_CASES = [
    ("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("MCMXCIV", 1994),
    ("MMMCMXCIX", 3999), ("", 0), ("IIII", 0), ("VV", 0), ("IC", 0),
    ("ABC", 0), ("MMMM", 0),
]


def verify(answer: str) -> float:
    """Score generated code on the test cases (0.0 if it does not run).

    NOTE: this executes model-generated code, exactly like the notebook does.
    """
    code = answer.strip().removeprefix("```python").removeprefix("```").removesuffix("```")
    try:
        namespace = {}
        exec(code, namespace)
        fn = namespace["roman_to_int"]
        return sum(fn(inp) == expected for inp, expected in TEST_CASES) / len(TEST_CASES)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    # Gemma 3 12B (no native thinking); temperature 1 for diverse samples
    llm = LLM(model=config.model("prompt"), temperature=1)

    print("== 1. Few-shot (answer only)")
    messages = [
        {"role": "user", "content": "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see? Be concise."},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": PENGUINS},
    ]
    print(llm.generate(messages).content, "\n")

    print("== 2. Chain-of-Thought (few-shot with reasoning)")
    messages = [
        {"role": "user", "content": "I saw 6 flamingos. 2 flew away. 1 hid behind a tree. How many can I see?"},
        {"role": "assistant", "content": "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. You see now 4 - 1 = 3."},
        {"role": "user", "content": PENGUINS},
    ]
    print(llm.generate(messages).content, "\n")

    print("== 3. Zero-shot Chain-of-Thought")
    messages = [{"role": "user", "content": PENGUINS + " Let's think step by step."}]
    print(llm.generate(messages).content, "\n")

    print(f"== 4. Self-consistency ({args.samples} samples, correct answer: 5)")
    answers = []
    for _ in range(args.samples):
        response = llm.generate([{"role": "user", "content": SEATING}])
        answers.append((response.content or "").split("Answer:")[-1].replace("\n", "").strip())
    print(Counter(answers), "\n")

    print(f"== 5. Best-of-N ({args.samples} samples)")
    responses = []
    for _ in range(args.samples):
        response = llm.generate([{"role": "user", "content": ROMAN}])
        responses.append((response.content or "", verify(response.content or "")))
    best_answer, best_score = max(responses, key=lambda x: x[1])
    print("Scores:", [score for _, score in responses])
    print(f"Best ({best_score:.2f}):\n{best_answer}\n")

    print("== 6. Native reasoning")
    thinker = LLM(model=config.model("native"), think=True)
    response = thinker.generate([{"role": "user", "content": PENGUINS}])
    print("REASONING:", response.reasoning)
    print("ANSWER:   ", response.content)


if __name__ == "__main__":
    main()
