"""Command line: ask "question"."""

import argparse
import sys

from research_assistant.llm import IncompleteResponseError, ask


def main() -> None:
    parser = argparse.ArgumentParser(prog="ask", description="Ask a company research question.")
    parser.add_argument("question", help='e.g. "What does Stripe sell, and to whom?"')
    args = parser.parse_args()

    try:
        print(ask(args.question))
    except IncompleteResponseError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
