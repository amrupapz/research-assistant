"""Run evals. Retrieval and answers are scored separately.

  uv run python evals/run.py --stage retrieval      # cheap: one embedding per question
  uv run python evals/run.py --stage answers        # ~1 model call per question
  uv run python evals/run.py --stage answers --only q07b --repeat 5   # measure a flaky case
  uv run python evals/run.py --stage qualify        # lead qualification vs hand labels (D6 M2), ~1 call per case
  uv run python evals/run.py --stage judge          # calibrate the claim judge vs MY labels (D6 M3)
  uv run python evals/run.py --stage answers --judge    # + judge every cited claim: unsupported fails, partial warns
                                                        # (--all always judges; single stages only with --judge)
  uv run python evals/run.py --stage agent --repeat 3   # agent trajectories, ~4-6 model calls per run
  uv run python evals/run.py --stage flow               # the same cases through the LangGraph flow (D5 M5)
  uv run python evals/run.py --compare                  # latest agent vs flow result per case, side by side
  uv run python evals/run.py --all                      # every stage in order → one summary table (D6 M0)

Each case appends one JSON line to runs/evals.jsonl (tagged with stage, category + PROMPT_VERSION).
After any run: pass rates per category (D6 M2), the numbers M5's baseline and regression gate compare.
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

from research_assistant import flow, tracing
from research_assistant.agent import (
    MaxStepsError,
    ResumeError,
    SimulatedCrash,
    resume_agent,
    run_agent,
)
from research_assistant.config import get_settings
from research_assistant.db import connect, fetch_lead, seed_leads
from research_assistant.evaluate import (
    UNGRADED,
    agreement,
    apply_claim_checks,
    claim_warnings,
    grade_agent_run,
    grade_answer,
    grade_flow_run,
    grade_qualification,
    load_agent_cases,
    load_cases,
    load_judge_labels,
    load_qualify_cases,
    observations,
    retrieval_scores,
    split_claims,
)
from research_assistant.llm import (
    IncompleteResponseError,
    UngroundedAnswerError,
    ask,
    embed,
    judge_claim,
)
from research_assistant.prompts import (
    AGENT_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    PROMPT_VERSION,
)
from research_assistant.retrieval import (
    fetch_chunks,
    hybrid_search,
    keyword_search,
    vector_search,
)
from research_assistant.schemas import Lead
from research_assistant.sending import send_once
from research_assistant.state import (
    get_steps,
    insert_draft,
    list_drafts,
    record_approval,
    record_reply,
)
from research_assistant.tools import QualifyLeadArgs, ToolContext, qualify_lead

ROOT = Path(__file__).parent.parent
CASES = ROOT / "evals" / "questions.jsonl"
AGENT_CASES = ROOT / "evals" / "agent_cases.jsonl"
QUALIFY_CASES = ROOT / "evals" / "qualify_cases.jsonl"
JUDGE_LABELS = ROOT / "evals" / "judge_labels.jsonl"
OUT = ROOT / "runs" / "evals.jsonl"
KS = (1, 3, 5)


RESULTS: list[dict] = []  # {"stage", "category", "pass"} for every graded run in this process


def _graded(stage: str, category: str, ok: bool) -> None:
    RESULTS.append({"stage": stage, "category": category, "pass": bool(ok)})


def category_rows(results: list[dict]) -> list[dict]:
    """Pass counts per category (rows) and stage (columns). Pure, so it's testable."""
    rows: dict[str, dict] = {}
    for r in results:
        row = rows.setdefault(r["category"], {"category": r["category"], "passed": 0, "total": 0, "stages": {}})
        cell = row["stages"].setdefault(r["stage"], [0, 0])
        cell[0] += r["pass"]
        cell[1] += 1
        row["passed"] += r["pass"]
        row["total"] += 1
    return sorted(rows.values(), key=lambda row: row["category"])


def category_table(rows: list[dict], stage_names: list[str]) -> str:
    lines = [f"{'category':<23}" + "".join(f"{s:>10}" for s in stage_names) + f"{'total':>10}{'rate':>7}"]
    for row in rows:
        cells = ["{}/{}".format(*row["stages"][s]) if s in row["stages"] else "-" for s in stage_names]
        total = f"{row['passed']}/{row['total']}"
        rate = f"{row['passed'] / row['total']:.0%}"
        lines.append(f"{row['category']:<23}" + "".join(f"{c:>10}" for c in cells) + f"{total:>10}{rate:>7}")
    return "\n".join(lines)


BATCH = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")  # one per process: the report groups records by it (M4)


def _write(record: dict) -> None:
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as f:
        f.write(
            json.dumps(
                {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "batch": BATCH, **record}
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


def run_retrieval(conn) -> tuple[int, int]:
    """Returns (hits, cases) for hybrid@5: the mode and k the app actually uses."""
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
    return hits[("hybrid", 5)], len(cases)


def judged_claims(claims, chunks_for) -> list[dict]:
    """Judge each claim that has evidence to check against. How verdicts count is decided by
    evaluate.apply_claim_checks (M3 option 1), not here.
    chunks_for(claim) -> the chunks to judge it against ([] = uncited: listed, not judged)."""
    out = []
    for claim in claims:
        chunks = chunks_for(claim)
        row = {"text": claim.text, "chunk_ids": [c.chunk_id for c in chunks], "verdict": "uncited", "reason": None}
        if chunks:
            try:
                v = judge_claim(claim.text, chunks)
                row.update(verdict=v.verdict, reason=v.reason)
            except Exception as e:  # noqa: BLE001 -- an observation failing must not fail the case
                row.update(verdict="judge_error", reason=f"{type(e).__name__}: {e}"[:200])
        out.append(row)
    return out


def _claims_line(claims: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for c in claims:
        counts[c["verdict"]] += 1
    flagged = [c for c in claims if c["verdict"] in ("unsupported", "partial", "judge_error")]
    line = "      claims (observed): " + ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
    return line + "".join(f"\n        ⚑ {c['verdict']}: {c['text'][:110]} — {c['reason']}" for c in flagged)


def run_answers(
    conn, k: int = 5, only: list[str] | None = None, repeat: int = 1, judge: bool = False
) -> tuple[int, int]:
    cases = [c for c in load_cases(CASES) if not only or c.id in only] * repeat
    passed = 0
    for case in cases:
        with tracing.eval_trace("answers", case.id, case.category) as root:  # retrieval + ask + judge: ONE trace
            chunks = [c for c, _ in hybrid_search(conn, case.question, k)]
            result, error = None, None
            try:
                result = ask(case.question, chunks)
            except (IncompleteResponseError, ValidationError, UngroundedAnswerError) as e:
                error = f"{type(e).__name__}: {e}"
            claims = []
            if judge and result is not None:
                by_id = {c.chunk_id: c for c in chunks}
                claims = judged_claims(
                    split_claims(result.answer), lambda cl, b=by_id: [b[i] for i in cl.chunk_ids if i in b]
                )
        trace_ids = [root.trace_id] if root.trace_id else []
        checks = apply_claim_checks(grade_answer(case, result), claims, graded=True)  # unsupported fails (M3)
        passed += checks["pass"]
        _graded("answers", case.category, checks["pass"])
        failed = [name for name, ok in checks.items() if not ok and name != "pass"]
        print(
            f"{'PASS' if checks['pass'] else 'FAIL'}  {case.id:<5} {case.expect:<8} {', '.join(failed)}"
        )
        if not checks["pass"]:
            print(f"      ↳ {(result.answer if result else error)[:160]}")
        if claims:
            print(_claims_line(claims))
        _write(
            {
                "stage": "answers",
                "prompt_version": PROMPT_VERSION,
                "case": case.id,
                "category": case.category,
                "trace_ids": trace_ids,
                "expect": case.expect,
                **checks,
                "error": error,
                "answer": result.answer if result else None,
                "needs_sources": result.needs_sources if result else None,
                "cited_ids": [c.chunk_id for c in result.citations] if result else [],
                "retrieved_ids": [c.chunk_id for c in chunks],
                "claims": claims,  # unsupported -> no_unsupported_claims=False; partial -> warning only
                "claim_warnings": claim_warnings(claims),
            }
        )
    print(f"\n{passed}/{len(cases)} passed (prompt {PROMPT_VERSION})")
    return passed, len(cases)


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


def _research_chunk_ids(conn, run_id) -> list[str]:
    """Every chunk cited by the run's successful search_company answers, in first-seen order."""
    ids = [
        c["chunk_id"]
        for s in get_steps(conn, run_id)
        if s["kind"] == "tool_call" and s["tool"] == "search_company" and s["status"] == "ok"
        for c in (s["output"] or {}).get("citations", [])
    ]
    return list(dict.fromkeys(ids))


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
        research_ids = _research_chunk_ids(conn, ctx.run_id) if ctx.run_id else []
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
        "research_ids": research_ids,  # chunks the run's search_company answers cited (drafts are judged against these)
        "trace_ids": [t for t in [tracing.trace_id_for(ctx.run_id) if ctx.run_id else None] if t],
    }


def run_agent_evals(conn, only: list[str] | None = None, repeat: int = 1, judge: bool = False) -> tuple[int, int]:
    cases = [c for c in load_agent_cases(AGENT_CASES) if not only or c.id in only]
    tally: dict[str, list[bool]] = defaultdict(list)
    seen: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _ in range(repeat):
        for case in cases:
            run = run_agent_case(conn, case)
            checks = grade_agent_run(case.expect, run)
            obs = observations(case, run)
            tally[case.id].append(checks["pass"])
            _graded("agent", case.category, checks["pass"])
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
            claims = []
            if judge and run["drafts"]:
                research = fetch_chunks(conn, run["research_ids"])  # drafts carry no citations: check vs the research
                with tracing.eval_trace("judge", case.id, case.category) as jroot:  # grader cost: its own trace
                    for d in run["drafts"]:
                        claims += judged_claims(split_claims(d["body"]), lambda _cl, r=research: r)
                run["trace_ids"] += [jroot.trace_id] if jroot.trace_id else []
                print(_claims_line(claims))
            _write(
                {
                    "stage": "agent",
                    "prompt_version": AGENT_PROMPT_VERSION,
                    "case": case.id,
                    "category": case.category,
                    "trace_ids": run["trace_ids"],
                    **checks,
                    "observed": obs,
                    "trail": trail,
                    "error": run["error"],
                    "answer": run["answer"],
                    "drafts": run["drafts"],
                    "sent": run["sent"],
                    "asked": run["asked"],
                    "claims": claims,  # drafts: warnings only (judge not calibrated on lead facts from our tables)
                    "claim_warnings": claim_warnings(claims),
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
    return sum(total), len(total)


def show_judge_items(conn, only: list[str] | None = None) -> None:
    """For labelling: each claim with the EXACT chunk text it cites (chunks are parts of pages, so
    the data/ file alone can mislead). Prints no verdicts and makes no model calls."""
    for item in [i for i in load_judge_labels(JUDGE_LABELS) if not only or i.id in only]:
        print(f"\n=== {item.id}  label={item.label}  ({item.origin})\nCLAIM: {item.claim}")
        for c in fetch_chunks(conn, item.chunk_ids) or []:
            print(f"--- {c.chunk_id}\n{c.text}")
        if not item.chunk_ids:
            print("--- (no chunks cited)")


def run_judge_calibration(conn, only: list[str] | None = None) -> tuple[int, int]:
    """The judge vs MY hand labels. Refuses to run while any label is missing, so I label blind
    (before seeing a single verdict). Returns (agreements, labelled claims)."""
    items = [i for i in load_judge_labels(JUDGE_LABELS) if not only or i.id in only]
    missing = [i.id for i in items if i.label is None]
    if missing:
        print(f"Label these first, before seeing any verdict: {', '.join(missing)}\n(evals/judge_labels.jsonl)")
        return 0, 0
    pairs, by_split = [], defaultdict(list)
    for item in items:
        try:
            v = judge_claim(item.claim, fetch_chunks(conn, item.chunk_ids))
            verdict, reason = v.verdict, v.reason
        except Exception as e:  # noqa: BLE001 -- a judge that errors disagrees
            verdict, reason = "judge_error", f"{type(e).__name__}: {e}"[:200]
        pairs.append((item.label, verdict))
        by_split[item.split].append((item.label, verdict))
        mark = "✓" if verdict == item.label else "✗"
        print(f"{mark} {item.id:<4} {item.split:<8} mine={item.label:<12} judge={verdict:<12} {item.claim[:62]}")
        if verdict != item.label:
            print(f"      ↳ {reason}")
        _write({"stage": "judge", "case": item.id, "split": item.split, "prompt_version": JUDGE_PROMPT_VERSION,
                "label": item.label, "verdict": verdict, "reason": reason})
    a = agreement(pairs)
    a["provisional"] = sum(i.labelled_by != "me" for i in items)  # labels a human hasn't reviewed
    labels = sorted({lab for lab, _ in pairs})
    verdicts = sorted({v for _, v in pairs} | set(labels))
    print(f"\n{'mine \\ judge':<14}" + "".join(f"{v:>13}" for v in verdicts))
    for lab in labels:
        print(f"{lab:<14}" + "".join(f"{a['confusion'][lab].get(v, 0):>13}" for v in verdicts))
    print(f"\njudge prompt {JUDGE_PROMPT_VERSION}")
    for name, sub in [*sorted(by_split.items()), ("all", pairs)]:
        b = agreement(sub)
        print(
            f"{name:<8} agreement {b['agree']}/{b['n']} = {b['rate']:.0%}  |  unsupported missed: {b['missed_unsupported']}"
            f"  |  bar (>=85%, 0 missed): {'PASSED' if b['passes_bar'] else 'NOT passed'}"
        )
        a[f"split_{name}"] = b
    if "holdout" in by_split:
        print("holdout = the honest number. Don't tune the prompt on it and re-run: write NEW held-out claims.")
    if a["provisional"]:
        print(
            f"⚠ PROVISIONAL: {a['provisional']}/{a['n']} labels are not reviewed by me (labelled_by != 'me'). "
            "Review them, set labelled_by to 'me', re-run: only then is this calibration."
        )
    _write({"stage": "judge_summary", **a})
    return a["agree"], a["n"]


def run_qualify_case(conn, case) -> dict:
    """The real qualify_lead (retrieval + judge_icp + code), same isolation as run_agent_case.

    reuse_qualification=False: always judge fresh (a stored verdict would test nothing)."""
    conn.execute("SELECT 1")
    conn.execute(f"TRUNCATE {AGENT_TABLES} RESTART IDENTITY")
    result, error = None, None
    with measured() as usage:
        if case.lead is not None:
            email = f"{case.lead_id.lower()}@example.com"
            seed_leads(conn, [Lead(lead_id=case.lead_id, email=email, **case.lead.model_dump())])
        ctx = ToolContext(conn=conn, reuse_qualification=False)
        with tracing.eval_trace("qualify", case.id, case.category) as root:
            try:
                result = qualify_lead(QualifyLeadArgs(lead_id=case.lead_id), ctx)
            except Exception as e:  # noqa: BLE001 -- report it as a failed case, go on
                error = f"{type(e).__name__}: {e}"
    conn.rollback()  # also removes the synthetic lead
    return {"result": result, "error": error, "usage": usage, "trace_ids": [root.trace_id] if root.trace_id else []}


def run_qualify_evals(conn, only: list[str] | None = None, repeat: int = 1) -> tuple[int, int]:
    cases = [c for c in load_qualify_cases(QUALIFY_CASES) if not only or c.id in only] * repeat
    passed = 0
    for case in cases:
        run = run_qualify_case(conn, case)
        checks = grade_qualification(case, run["result"])
        passed += checks["pass"]
        _graded("qualify", case.category, checks["pass"])
        q = run["result"] or {}
        verdicts = " ".join(f"{c['id']}={c['met']}" for c in q.get("criteria", []))
        failed = [n for n, ok in checks.items() if not ok and n != "pass" and not n.startswith(UNGRADED)]
        notes = [n for n, ok in checks.items() if not ok and n.startswith(UNGRADED)]
        print(
            f"{'PASS' if checks['pass'] else 'FAIL'}  {case.id}  fit={q.get('fit', '-')}  {verdicts}"
            f"  {', '.join(failed)}{'  (not graded: ' + ', '.join(notes) + ')' if notes else ''}",
            flush=True,
        )
        if run["error"]:
            print(f"      ↳ {run['error'][:160]}")
        _write(
            {
                "stage": "qualify",
                "case": case.id,
                "category": case.category,
                "trace_ids": run["trace_ids"],
                **checks,
                "error": run["error"],
                "fit": q.get("fit"),
                "criteria": q.get("criteria"),
                **run["usage"],
            }
        )
    print(f"\n{passed}/{len(cases)} passed (labels: evals/qualify_cases.jsonl)")
    return passed, len(cases)


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
    trace_ids = [t for t in [tracing.trace_id_for(ctx.run_id) if ctx.run_id else None] if t]
    return {"outcome": outcome, "error": error, "drafts": drafts, "sent": sent, "usage": usage, "trace_ids": trace_ids}


def run_flow_evals(conn, only: list[str] | None = None, repeat: int = 1) -> tuple[int, int]:
    cases = [c for c in load_agent_cases(AGENT_CASES) if c.flow and (not only or c.id in only)]
    tally: dict[str, list[bool]] = defaultdict(list)
    for _ in range(repeat):
        for case in cases:
            run = run_flow_case(conn, case)
            checks = grade_flow_run(case.flow.expect, run)
            tally[case.id].append(checks["pass"])
            _graded("flow", case.category, checks["pass"])
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
                    "category": case.category,
                    "trace_ids": run["trace_ids"],
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
    return sum(total), len(total)


def stages(conn, only: list[str] | None = None, repeat: int = 1, judge: bool = False) -> dict:
    """Every stage, in order: cheap and deterministic first, so a broken retrieval shows up before
    you pay for ~170 agent/flow calls. Each returns (passed, total)."""
    return {
        "retrieval": lambda: run_retrieval(conn),  # hybrid@5; --only/--repeat don't apply
        "answers": lambda: run_answers(conn, only=only, repeat=repeat, judge=judge),
        "qualify": lambda: run_qualify_evals(conn, only=only, repeat=repeat),
        "agent": lambda: run_agent_evals(conn, only=only, repeat=repeat, judge=judge),
        "flow": lambda: run_flow_evals(conn, only=only, repeat=repeat),
    }


def run_all(stage_fns: dict, meter=measured) -> list[dict]:
    """Run each stage under the meter; one row per stage. A stage that crashes is reported and
    the rest still run (one bad stage shouldn't hide the others)."""
    rows = []
    for name, fn in stage_fns.items():
        print(f"\n=== {name} ===", flush=True)
        with meter() as usage:
            try:
                passed, total = fn()
                error = None
            except Exception as e:  # noqa: BLE001 -- report it in the table, keep going
                passed, total, error = 0, 0, f"{type(e).__name__}: {e}"
        rows.append({"stage": name, "passed": passed, "total": total, "error": error, **usage})
    return rows


def summary_table(rows: list[dict]) -> str:
    lines = [f"{'stage':<11}{'passed':>10}{'rate':>7}{'calls':>7}{'tokens':>9}{'seconds':>9}"]
    for r in rows:
        rate = f"{r['passed'] / r['total']:.0%}" if r["total"] else "-"
        tokens = r.get("input_tokens", 0) + r.get("output_tokens", 0)
        lines.append(
            f"{r['stage']:<11}{f'{r["passed"]}/{r["total"]}':>10}{rate:>7}"
            f"{r.get('model_calls', 0):>7}{tokens:>9}{r.get('seconds', 0):>9}"
            + (f"   ERROR {r['error'][:60]}" if r["error"] else "")
        )
    return "\n".join(lines)


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
    parser.add_argument("--stage", choices=["retrieval", "answers", "qualify", "agent", "flow", "judge"])
    parser.add_argument("--compare", action="store_true", help="agent vs flow, from runs/evals.jsonl")
    parser.add_argument("--all", action="store_true", help="every stage in order, then one summary table")
    parser.add_argument(
        "--only", nargs="*", help="case ids to run, e.g. --only q07 q07b"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="run each case N times (flakiness)"
    )
    parser.add_argument("--judge", action="store_true", help="judge claims (D6 M3): unsupported fails an answer; partial and drafts only warn")
    parser.add_argument("--show", action="store_true", help="with --stage judge: print claims + cited chunks to label")
    args = parser.parse_args()
    if args.compare:
        return compare()
    if args.all:
        with connect() as conn:
            rows = run_all(stages(conn, only=args.only, repeat=args.repeat, judge=True))  # the full run always judges
        print("\n" + summary_table(rows))
        cats = _print_categories()
        _write({"stage": "all", "rows": rows, "categories": cats})
        return
    if not args.stage:
        parser.error("give --stage, --all or --compare")
    with connect() as conn:
        if args.stage == "retrieval":
            run_retrieval(conn)
        elif args.stage == "agent":
            run_agent_evals(conn, only=args.only, repeat=args.repeat, judge=args.judge)
        elif args.stage == "flow":
            run_flow_evals(conn, only=args.only, repeat=args.repeat)
        elif args.stage == "qualify":
            run_qualify_evals(conn, only=args.only, repeat=args.repeat)
        elif args.stage == "judge":
            show_judge_items(conn, args.only) if args.show else run_judge_calibration(conn, args.only)
        else:
            run_answers(conn, only=args.only, repeat=args.repeat, judge=args.judge)
    _print_categories()


def _print_categories() -> list[dict]:
    """The per-category table for everything graded in this process (retrieval isn't behaviour: excluded)."""
    rows = category_rows(RESULTS)
    if rows:
        stage_names = [s for s in ("answers", "qualify", "agent", "flow") if any(r["stage"] == s for r in RESULTS)]
        print("\n" + category_table(rows, stage_names))
    return rows


if __name__ == "__main__":
    try:
        main()
    finally:
        tracing.shutdown()  # D6 M1: eval runs are traced too when TRACING_ENABLED; send before exit
