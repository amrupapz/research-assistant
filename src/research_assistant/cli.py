"""Command line: ask "question" [-k 5] [--json]. Hybrid retrieval -> grounded answer."""

import argparse
import sys

from pydantic import ValidationError

from research_assistant import tracing
from research_assistant.llm import IncompleteResponseError, UngroundedAnswerError, ask
from research_assistant.schemas import Chunk, ResearchAnswer


def format_answer(result: ResearchAnswer, chunks: list[Chunk]) -> str:
    lines = [
        result.answer,
        "",
        f"confidence: {result.confidence} | needs sources: {'yes' if result.needs_sources else 'no'}",
    ]
    if result.assumptions:
        lines.append("assumptions:")
        lines += [f"  - {a}" for a in result.assumptions]
    fetched = {c.chunk_id: c.fetched_at for c in chunks}
    if result.citations:
        lines.append("sources:")
        lines += [f"  [{c.chunk_id}] {c.source_url} (fetched {fetched.get(c.chunk_id)})" for c in result.citations]
    return "\n".join(lines)


def main() -> None:
    from research_assistant.db import connect
    from research_assistant.retrieval import hybrid_search

    parser = argparse.ArgumentParser(prog="ask", description="Ask a company research question.")
    parser.add_argument("question", help='e.g. "Who is the current CEO of Intel?"')
    parser.add_argument("-k", type=int, default=5, help="chunks to retrieve (default 5)")
    parser.add_argument("--json", action="store_true", help="print the raw structured answer")
    args = parser.parse_args()

    try:
        with tracing.observe("ask", as_type="chain", input={"question": args.question, "k": args.k}) as span:
            with connect() as conn:
                chunks = [chunk for chunk, _ in hybrid_search(conn, args.question, k=args.k)]
            result = ask(args.question, chunks)
            span.update(output=result.answer)
    except (IncompleteResponseError, ValidationError, UngroundedAnswerError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        tracing.shutdown()

    print(result.model_dump_json(indent=2) if args.json else format_answer(result, chunks))


if __name__ == "__main__":
    main()
