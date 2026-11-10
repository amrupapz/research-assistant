"""Command line: ask "question" [--json]."""

import argparse
import sys

from pydantic import ValidationError

from research_assistant.llm import IncompleteResponseError, ask
from research_assistant.schemas import ResearchAnswer


def format_answer(result: ResearchAnswer) -> str:
    lines = [
        result.answer,
        "",
        f"confidence: {result.confidence} | needs sources: {'yes' if result.needs_sources else 'no'}",
    ]
    if result.assumptions:
        lines.append("assumptions:")
        lines += [f"  - {a}" for a in result.assumptions]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ask", description="Ask a company research question.")
    parser.add_argument("question", help='e.g. "What does Stripe sell, and to whom?"')
    parser.add_argument("--json", action="store_true", help="print the raw structured answer")
    args = parser.parse_args()

    try:
        result = ask(args.question)
    except (IncompleteResponseError, ValidationError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result.model_dump_json(indent=2) if args.json else format_answer(result))


if __name__ == "__main__":
    main()
