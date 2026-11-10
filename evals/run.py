"""Run evals. Retrieval and answers are scored separately.

  uv run python evals/run.py --stage retrieval      # cheap: one embedding per question
  uv run python evals/run.py --stage answers        # ~1 model call per question
  uv run python evals/run.py --stage answers --only q07b --repeat 5   # measure a flaky case

Each case appends one JSON line to runs/evals.jsonl (tagged with stage + PROMPT_VERSION).
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from research_assistant.db import connect
from research_assistant.evaluate import grade_answer, load_cases, retrieval_scores
from research_assistant.llm import (
    IncompleteResponseError,
    UngroundedAnswerError,
    ask,
    embed,
)
from research_assistant.prompts import PROMPT_VERSION
from research_assistant.retrieval import hybrid_search, keyword_search, vector_search

ROOT = Path(__file__).parent.parent
CASES = ROOT / "evals" / "questions.jsonl"
OUT = ROOT / "runs" / "evals.jsonl"
KS = (1, 3, 5)


def _write(record: dict) -> None:
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record}) + "\n")


def memo_embed():
    """Embed each distinct query once, even though vector + hybrid both need it."""
    cache: dict[str, list[float]] = {}

    def _embed(texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in cache]
        if missing:
            cache.update(zip(missing, embed(missing), strict=True))
        return [cache[t] for t in texts]

    return _embed


def run_retrieval(conn) -> None:
    emb = memo_embed()
    modes = {
        "keyword": lambda q, k: keyword_search(conn, q, k),
        "any": lambda q, k: keyword_search(conn, q, k, match="any"),
        "vector": lambda q, k: vector_search(conn, q, k, embed=emb),
        "hybrid": lambda q, k: hybrid_search(conn, q, k, embed=emb),
    }
    cases = [c for c in load_cases(CASES) if c.gold_sources]
    hits = {(m, k): 0 for m in modes for k in KS}
    print(f"{'case':<6}" + "".join(f"{m + '@' + str(k):>11}" for m in modes for k in KS))
    for case in cases:
        row = f"{case.id:<6}"
        for mode, search in modes.items():
            chunks = [c for c, _ in search(case.question, max(KS))]
            for k in KS:
                s = retrieval_scores(case, chunks[:k])
                hits[(mode, k)] += s["hit"]
                row += f"{'✅' if s['hit'] else f'f{s["file_recall"]:.1f}/x{s["fact_recall"]:.1f}':>11}"
                _write({"stage": "retrieval", "case": case.id, "mode": mode, "k": k, **s,
                        "retrieved_ids": [c.chunk_id for c in chunks[:k]]})
        print(row)
    print(f"{'hit%':<6}" + "".join(f"{hits[(m, k)] / len(cases):>11.0%}" for m in modes for k in KS))
    print("\n✅ = all gold files and facts in top-k; otherwise f=file_recall x=fact_recall")


def run_answers(conn, k: int = 5, only: list[str] | None = None, repeat: int = 1) -> None:
    cases = [c for c in load_cases(CASES) if not only or c.id in only] * repeat
    passed = 0
    for case in cases:
        chunks = [c for c, _ in hybrid_search(conn, case.question, k)]
        result, error = None, None
        try:
            result = ask(case.question, chunks)
        except (IncompleteResponseError, ValidationError, UngroundedAnswerError) as e:
            error = f"{type(e).__name__}: {e}"
        checks = grade_answer(case, result)
        passed += checks["pass"]
        failed = [name for name, ok in checks.items() if not ok and name != "pass"]
        print(f"{'PASS' if checks['pass'] else 'FAIL'}  {case.id:<5} {case.expect:<8} {', '.join(failed)}")
        if not checks["pass"]:
            print(f"      ↳ {(result.answer if result else error)[:160]}")
        _write({"stage": "answers", "prompt_version": PROMPT_VERSION, "case": case.id, "expect": case.expect,
                **checks, "error": error, "answer": result.answer if result else None,
                "needs_sources": result.needs_sources if result else None,
                "cited_ids": [c.chunk_id for c in result.citations] if result else [],
                "retrieved_ids": [c.chunk_id for c in chunks]})
    print(f"\n{passed}/{len(cases)} passed (prompt {PROMPT_VERSION})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retrieval", "answers"], required=True)
    parser.add_argument("--only", nargs="*", help="case ids to run, e.g. --only q07 q07b")
    parser.add_argument("--repeat", type=int, default=1, help="run each answer case N times (flakiness)")
    args = parser.parse_args()
    with connect() as conn:
        if args.stage == "retrieval":
            run_retrieval(conn)
        else:
            run_answers(conn, only=args.only, repeat=args.repeat)


if __name__ == "__main__":
    main()
