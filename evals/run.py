"""Run evals. Retrieval and answers are scored separately.

  uv run python evals/run.py --stage retrieval      # cheap: one embedding per question
  uv run python evals/run.py --stage answers        # ~1 model call per question
  uv run python evals/run.py --stage answers --only q07b --repeat 5   # measure a flaky case
  uv run python evals/run.py --stage agent --repeat 3   # agent trajectories, ~4-6 model calls per run
  uv run python evals/run.py --stage flow               # the same cases through the LangGraph flow (D5 M5)
  uv run python evals/run.py --compare                  # latest agent vs flow result per case, side by side

Each case appends one JSON line to runs/evals.jsonl (tagged with stage + PROMPT_VERSION).
"""

import argparse
import json
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from research_assistant import flow
from research_assistant.agent import (
    MaxStepsError,
    ResumeError,
    SimulatedCrash,
    resume_agent,
    run_agent,
)
from research_assistant.config import get_settings
from research_assistant.db import connect, fetch_lead
from research_assistant.evaluate import (
    grade_agent_run,
    grade_answer,
    grade_flow_run,
    load_agent_cases,
    load_cases,
    observations,
    retrieval_scores,
)
from research_assistant.llm import (
    IncompleteResponseError,
    UngroundedAnswerError,
    ask,
    embed,
)
from research_assistant.prompts import AGENT_PROMPT_VERSION, PROMPT_VERSION
from research_assistant.retrieval import hybrid_search, keyword_search, vector_search
from research_assistant.sending import send_once
from research_assistant.state import (
    get_steps,
    insert_draft,
    list_drafts,
    record_approval,
    record_reply,
)
from research_assistant.tools import ToolContext

ROOT = Path(__file__).parent.parent
CASES = ROOT / "evals" / "questions.jsonl"
AGENT_CASES = ROOT / "evals" / "agent_cases.jsonl"
OUT = ROOT / "runs" / "evals.jsonl"
KS = (1, 3, 5)


def _write(record: dict) -> None:
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as f:
        f.write(
            json.dumps(
                {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record}
            )
            + "\n"
        )


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
    print(
        f"{'case':<6}" + "".join(f"{m + '@' + str(k):>11}" for m in modes for k in KS)
    )
    for case in cases:
        row = f"{case.id:<6}"
        for mode, search in modes.items():
            chunks = [c for c, _ in search(case.question, max(KS))]
            for k in KS:
                s = retrieval_scores(case, chunks[:k])
                hits[(mode, k)] += s["hit"]
                row += f"{'✅' if s['hit'] else f'f{s["file_recall"]:.1f}/x{s["fact_recall"]:.1f}':>11}"
                _write(
                    {
                        "stage": "retrieval",
                        "case": case.id,
                        "mode": mode,
                        "k": k,
                        **s,
                        "retrieved_ids": [c.chunk_id for c in chunks[:k]],
                    }
                )
        print(row)
    print(
        f"{'hit%':<6}"
        + "".join(f"{hits[(m, k)] / len(cases):>11.0%}" for m in modes for k in KS)
    )
    print(
        "\n✅ = all gold files and facts in top-k; otherwise f=file_recall x=fact_recall"
    )


def run_answers(
    conn, k: int = 5, only: list[str] | None = None, repeat: int = 1
) -> None:
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
        print(
            f"{'PASS' if checks['pass'] else 'FAIL'}  {case.id:<5} {case.expect:<8} {', '.join(failed)}"
        )
        if not checks["pass"]:
            print(f"      ↳ {(result.answer if result else error)[:160]}")
        _write(
            {
                "stage": "answers",
                "prompt_version": PROMPT_VERSION,
                "case": case.id,
                "expect": case.expect,
                **checks,
                "error": error,
                "answer": result.answer if result else None,
                "needs_sources": result.needs_sources if result else None,
                "cited_ids": [c.chunk_id for c in result.citations] if result else [],
                "retrieved_ids": [c.chunk_id for c in chunks],
            }
        )
    print(f"\n{passed}/{len(cases)} passed (prompt {PROMPT_VERSION})")


AGENT_TABLES = "runs, run_steps, drafts, approvals, sends, qualifications, research_findings, finding_citations, replies"


def apply_setup(conn, setup, outbox: Path) -> None:
    """Past events for a case: emails sent N days ago (approved + send_once, then backdated), replies."""
    for e in setup.sent:
        draft = insert_draft(
            conn,
            fetch_lead(conn, e.lead_id),
            "Quick introduction",
            "Hello, a short intro.",
            None,
        )
        record_approval(conn, draft.draft_id, True, None)
        send_once(conn, draft, outbox)
        conn.execute(
            "UPDATE sends SET sent_at = now() - make_interval(days => %s) WHERE draft_id = %s",
            [e.days_ago, int(draft.draft_id[1:])],
        )
    for e in setup.replies:
        record_reply(conn, e.lead_id, "eval setup")
        conn.execute(
            "UPDATE replies SET received_at = now() - make_interval(days => %s) "
            "WHERE reply_id = (SELECT max(reply_id) FROM replies)",
            [e.days_ago],
        )


def _records_from_db(conn, run_id) -> list[dict]:
    """Tool-call records for the WHOLE run (both halves of a crash + resume), from run_steps."""
    return [
        {
            "step": s["turn"],
            "tool": s["tool"],
            "status": s["status"],
            "decision": s["decision"],
            "validated_args": (s["payload"] or {}).get("validated_args"),
        }
        for s in get_steps(conn, run_id)
        if s["kind"] == "tool_call"
    ]


@contextmanager
def measured():
    """Model calls / tokens / wall time while the block runs, read from the run-log lines it appends."""
    log = get_settings().runs_dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"  # (a case spanning midnight: undercounted)
    start_size = log.stat().st_size if log.exists() else 0
    start = time.perf_counter()
    usage: dict = {}
    yield usage
    new = log.read_text(encoding="utf-8")[start_size:] if log.exists() else ""  # ASCII JSON: chars == bytes
    calls = [r for r in map(json.loads, new.splitlines()) if r.get("kind") != "tool_call"]
    usage.update(
        model_calls=len(calls),
        input_tokens=sum(r.get("input_tokens") or 0 for r in calls),
        output_tokens=sum(r.get("output_tokens") or 0 for r in calls),
        seconds=round(time.perf_counter() - start, 1),
    )


def run_agent_case(conn, case) -> dict:
    """One real agent run with a TEMP outbox and a scripted approver. Returns plain data for grading.

    Everything (setup, run, resume) happens inside one outer transaction that is rolled back.
    """
    asked = []
    # Open an outer transaction: the run's steps become savepoints, rolled back below.
    conn.execute("SELECT 1")
    # A clean slate: the case sees leads + chunks + its own setup, never my real runs/sends/replies.
    # (Found in M7: a real reply from L1 in the dev DB made a12/a15/a16 "fail" correctly.)
    # TRUNCATE is transactional in Postgres, so the rollback below restores everything.
    conn.execute(f"TRUNCATE {AGENT_TABLES} RESTART IDENTITY")
    with tempfile.TemporaryDirectory() as tmp, measured() as usage:
        # Setup emails go to their own outbox: `sent` counts only the task's emails.
        apply_setup(conn, case.setup, Path(tmp) / "setup-outbox")
        ctx = ToolContext(conn=conn, outbox_dir=Path(tmp) / "outbox")

        def approver(call, _ctx):
            asked.append(call.name)
            return case.approve == "yes"

        answer, error, tool_calls = None, None, []
        try:
            try:
                result = run_agent(
                    case.task, ctx, approve_fn=approver, crash_after=case.crash_after
                )
            except SimulatedCrash:
                fresh = ToolContext(
                    conn=conn, outbox_dir=ctx.outbox_dir
                )  # "a new process"
                result = resume_agent(ctx.run_id, fresh, approve_fn=approver)
            answer = result.answer
        except (MaxStepsError, IncompleteResponseError, ResumeError) as e:
            error = f"{type(e).__name__}: {e}"
        tool_calls = _records_from_db(conn, ctx.run_id) if ctx.run_id else []
        sent = (
            sorted(f.name for f in ctx.outbox_dir.glob("*.eml"))
            if ctx.outbox_dir.exists()
            else []
        )
    drafts = (
        [d.model_dump() for d in list_drafts(conn, ctx.run_id)] if ctx.run_id else []
    )
    conn.rollback()  # evals leave no runs, drafts, sends or replies behind in the dev DB
    return {
        "tool_calls": tool_calls,
        "drafts": drafts,
        "sent": sent,
        "answer": answer,
        "error": error,
        "asked": len(asked),
        "usage": usage,
    }


def run_agent_evals(conn, only: list[str] | None = None, repeat: int = 1) -> None:
    cases = [c for c in load_agent_cases(AGENT_CASES) if not only or c.id in only]
    tally: dict[str, list[bool]] = defaultdict(list)
    seen: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _ in range(repeat):
        for case in cases:
            run = run_agent_case(conn, case)
            checks = grade_agent_run(case.expect, run)
            obs = observations(case, run)
            tally[case.id].append(checks["pass"])
            for o in obs:
                seen[case.id][o] += 1
            failed = [n for n, ok in checks.items() if not ok and n != "pass"]
            trail = " → ".join(f"{c['tool']}[{c['status']}]" for c in run["tool_calls"])
            print(
                f"{'PASS' if checks['pass'] else 'FAIL'}  {case.id}  {trail}  {', '.join(failed)}",
                flush=True,
            )
            if run["error"]:
                print(f"      ↳ {run['error'][:160]}")
            _write(
                {
                    "stage": "agent",
                    "prompt_version": AGENT_PROMPT_VERSION,
                    "case": case.id,
                    **checks,
                    "observed": obs,
                    "trail": trail,
                    "error": run["error"],
                    "answer": run["answer"],
                    "drafts": run["drafts"],
                    "sent": run["sent"],
                    "asked": run["asked"],
                    **run["usage"],
                }
            )
    print(f"\n{'case':<6}{'pass':>8}   observed in drafts (not graded)")
    for case in cases:
        runs = tally[case.id]
        obs = ", ".join(f"{k} {v}/{len(runs)}" for k, v in seen[case.id].items()) or "-"
        print(f"{case.id:<6}{sum(runs):>4}/{len(runs):<3}   {obs}")
    total = [ok for runs in tally.values() for ok in runs]
    print(f"\n{sum(total)}/{len(total)} runs passed (prompt {AGENT_PROMPT_VERSION})")


@contextmanager
def crash_after(tool: str | None):
    """The flow's version of an agent case's crash_after: die right after `tool` first returns."""
    if tool is None:
        yield
        return
    real, done = flow.call_tool, []

    def call_tool(ctx, name, args):
        out = real(ctx, name, args)
        if name == tool and not done:
            done.append(name)
            raise KeyboardInterrupt  # a crash: not an Exception, so guarded() lets it through
        return out

    flow.call_tool = call_tool
    try:
        yield
    finally:
        flow.call_tool = real


def run_flow_case(conn, case) -> dict:
    """One real flow run for a case, same isolation as run_agent_case (TRUNCATE + rollback).

    In-memory checkpointer: nothing commits. (Postgres checkpoints + a new process are covered by
    tests/test_flow.py; here a crash is resumed in the same process.)"""
    from langgraph.checkpoint.memory import InMemorySaver

    spec = case.flow
    conn.execute("SELECT 1")
    conn.execute(f"TRUNCATE {AGENT_TABLES} RESTART IDENTITY")
    saver = InMemorySaver()
    outcome, error = None, None
    with tempfile.TemporaryDirectory() as tmp, measured() as usage:
        apply_setup(conn, case.setup, Path(tmp) / "setup-outbox")
        ctx = ToolContext(conn=conn, outbox_dir=Path(tmp) / "outbox")
        try:
            try:
                with crash_after(case.crash_after):
                    final = flow.run_flow(spec.lead_id, ctx, checkpointer=saver)
            except KeyboardInterrupt:
                final = flow.resume_flow(ctx.run_id, ctx, saver)
            draft_id = (final.get("draft") or {}).get("draft_id")
            if spec.decide and draft_id and "approved" not in final:
                final = flow.approve_flow(draft_id, spec.decide == "yes", ctx, saver)
            outcome = final.get("outcome")
        except Exception as e:  # noqa: BLE001 -- transient errors that survived retries: report, go on
            error = f"{type(e).__name__}: {e}"
        sent = sorted(f.name for f in ctx.outbox_dir.glob("*.eml")) if ctx.outbox_dir.exists() else []
        drafts = [d.model_dump() for d in list_drafts(conn, ctx.run_id)] if ctx.run_id else []
    conn.rollback()
    return {"outcome": outcome, "error": error, "drafts": drafts, "sent": sent, "usage": usage}


def run_flow_evals(conn, only: list[str] | None = None, repeat: int = 1) -> None:
    cases = [c for c in load_agent_cases(AGENT_CASES) if c.flow and (not only or c.id in only)]
    tally: dict[str, list[bool]] = defaultdict(list)
    for _ in range(repeat):
        for case in cases:
            run = run_flow_case(conn, case)
            checks = grade_flow_run(case.flow.expect, run)
            tally[case.id].append(checks["pass"])
            failed = [n for n, ok in checks.items() if not ok and n != "pass"]
            u = run["usage"]
            print(
                f"{'PASS' if checks['pass'] else 'FAIL'}  {case.id}  {run['outcome'] or run['error']}"
                f"  ({u['model_calls']} calls, {u['seconds']}s)  {', '.join(failed)}",
                flush=True,
            )
            _write(
                {
                    "stage": "flow",
                    "prompt_version": flow.FLOW_VERSION,
                    "case": case.id,
                    **checks,
                    "outcome": run["outcome"],
                    "error": run["error"],
                    "drafts": run["drafts"],
                    "sent": run["sent"],
                    **u,
                }
            )
    skipped = [c.id for c in load_agent_cases(AGENT_CASES) if not c.flow]
    total = [ok for runs in tally.values() for ok in runs]
    print(f"\n{sum(total)}/{len(total)} runs passed ({flow.FLOW_VERSION}); not expressible as a flow: {', '.join(skipped)}")


def compare() -> None:
    """Latest agent and flow record per case, side by side (no API calls)."""
    latest: dict[tuple[str, str], dict] = {}
    for line in OUT.read_text(encoding="utf-8").splitlines() if OUT.exists() else []:
        r = json.loads(line)
        if r.get("stage") in ("agent", "flow"):
            latest[(r["stage"], r["case"])] = r

    def cell(r: dict | None) -> str:
        if r is None:
            return f"{'-':<28}"
        calls = r.get("model_calls", "?")
        tokens = (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) if "input_tokens" in r else "?"
        return f"{'PASS' if r['pass'] else 'FAIL'} {calls:>3} calls {tokens!s:>6} tok {r.get('seconds', '?')!s:>5}s"

    print(f"{'case':<6}{'agent (loop)':<34}{'flow (graph)':<34}")
    for case in load_agent_cases(AGENT_CASES):
        print(f"{case.id:<6}{cell(latest.get(('agent', case.id))):<34}{cell(latest.get(('flow', case.id))):<34}")
    print("(agent rows from before M5 have no calls/tokens: rerun --stage agent once)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retrieval", "answers", "agent", "flow"])
    parser.add_argument("--compare", action="store_true", help="agent vs flow, from runs/evals.jsonl")
    parser.add_argument(
        "--only", nargs="*", help="case ids to run, e.g. --only q07 q07b"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="run each case N times (flakiness)"
    )
    args = parser.parse_args()
    if args.compare:
        return compare()
    if not args.stage:
        parser.error("give --stage or --compare")
    with connect() as conn:
        if args.stage == "retrieval":
            run_retrieval(conn)
        elif args.stage == "agent":
            run_agent_evals(conn, only=args.only, repeat=args.repeat)
        elif args.stage == "flow":
            run_flow_evals(conn, only=args.only, repeat=args.repeat)
        else:
            run_answers(conn, only=args.only, repeat=args.repeat)


if __name__ == "__main__":
    main()
