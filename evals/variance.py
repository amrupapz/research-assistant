"""Ask the same question N times and see how much the structured answer varies.

Run: uv run python evals/variance.py [N]
(gpt-6-luna rejects `temperature`, so all runs use the model's fixed default.)
"""

import sys
from collections import Counter

from research_assistant.llm import ask

QUESTIONS = [
    ("stable", "What does Stripe sell, and to whom?"),
    ("unknowable", "How much did Anthropic raise in its latest round?"),
]


def run(question: str, n: int) -> None:
    print(f"\n### {question}  (x{n})")
    print(f"{'#':>2}  {'conf':<6} {'src':<5} {'len':>4}  answer")
    results = []
    for i in range(1, n + 1):
        r = ask(question, [])  # Day 1 experiment; since prompt v3 (strict grounding) this abstains
        results.append(r)
        flat = r.answer.replace("\n", " ")
        print(f"{i:>2}  {r.confidence:<6} {r.needs_sources!s:<5} {len(r.answer):>4}  {flat[:60]}")

    conf = Counter(r.confidence for r in results)
    src = Counter(r.needs_sources for r in results)
    lengths = [len(r.answer) for r in results]
    distinct = len({r.answer for r in results})
    print(f"confidence: {dict(conf)}  needs_sources: {dict(src)}")
    print(f"answer length: {min(lengths)}-{max(lengths)} chars, {distinct}/{n} distinct texts")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for _, question in QUESTIONS:
        run(question, n)


if __name__ == "__main__":
    main()
