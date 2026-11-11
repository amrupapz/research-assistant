"""Agent loop: model proposes -> validate -> authorize -> execute -> feed the result back. `agent "task"`.

D4 M3: every model turn and every tool call is a run_steps row. A tool call's effect (e.g. the
draft row), its approval row, and its step row commit in ONE transaction: all or nothing.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from research_assistant import state
from research_assistant.config import Settings, get_settings
from research_assistant.llm import (
    IncompleteResponseError,
    extract_function_calls,
    extract_output_text,
    function_call_output,
    make_client,
    respond,
)
from research_assistant.log import log_run
from research_assistant.permissions import Decision, ToolCall, authorize
from research_assistant.prompts import AGENT_PROMPT_VERSION
from research_assistant.qualify import fit_warning
from research_assistant.tools import (
    REGISTRY,
    ToolContext,
    ToolError,
    execute,
    search_warning,
    tool_definitions,
    validate_call,
)

RespondFn = Callable[[list[dict], list[dict]], dict]
AuthorizeFn = Callable[[ToolCall], Decision]
ApproveFn = Callable[[ToolCall, ToolContext], bool]


class MaxStepsError(Exception):
    """The model was still calling tools after max_steps turns."""


class ResumeError(Exception):
    """This run can't be resumed (unknown, already done, or started under a different prompt/model)."""


class SimulatedCrash(BaseException):
    """--crash-after: dies right after a tool's step commits. BaseException, like a real kill:
    `except Exception` doesn't catch it, so the run is NOT marked failed and stays 'running'."""


@dataclass
class AgentResult:
    answer: str
    steps: int  # model turns used
    tool_calls: list[dict]  # one log record per proposed call, in order
    run_id: UUID | None = None


def no_approver(call: ToolCall, ctx: ToolContext) -> bool:
    """Default when nobody is at the keyboard (tests, evals, batch): needs_approval means no."""
    return False


def _execute(name: str, args, ctx: ToolContext, record: dict, savepoint: bool = True) -> dict:
    """Run an authorized (and, if needed, approved) call; turn failures into {"error": ...}.

    Normally the tool runs inside a SAVEPOINT: if it fails halfway, its partial writes are undone,
    but the surrounding step transaction survives and still records the error. External tools
    (savepoint=False) manage their own commits.
    """
    try:
        if savepoint:
            with ctx.conn.transaction():
                output = execute(name, args, ctx)
        else:
            output = execute(name, args, ctx)
        record["status"] = "ok"
        return output
    except ToolError as e:
        record["status"] = "error"
        record["error"] = str(e)
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 -- any tool crash: log it all, tell the model little
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        return {"error": f"{name} failed ({type(e).__name__})"}


def handle_call(
    call: dict,
    ctx: ToolContext,
    authorize_fn: AuthorizeFn,
    step: int,
    settings: Settings,
    approve_fn: ApproveFn = no_approver,
) -> tuple[dict, dict]:
    """One proposed call through the gate: validate -> authorize -> (approve) -> execute. Always logs.

    Returns (what the model sees, the log record). Problems the model could fix (bad args,
    denied, not approved, tool error) become {"error": ...} outputs.
    If authorize_fn or approve_fn raises, the exception propagates and the run stops: fail closed.

    The human is asked BEFORE any transaction opens (never hold a transaction while waiting on a
    person). Then one transaction: the tool's effect + the approval row + the step row.
    """
    name = call["name"]
    record: dict = {
        "kind": "tool_call",
        "step": step,
        "call_id": call["call_id"],
        "tool": name,
        "arguments": call.get("arguments", "")[:500],
    }
    start = time.perf_counter()
    try:
        args, output = _decide(name, call.get("arguments", ""), ctx, authorize_fn, approve_fn, record)
        _save_approval(ctx, record)  # its own commit: send_once checks for it (M5)
        external = args is not None and REGISTRY[name].external
        if external:  # e.g. send_email: commits its own ledger steps, so it can't sit inside ours
            output = _execute(name, args, ctx, record, savepoint=False)
        with ctx.conn.transaction():
            if args is not None and not external:
                output = _execute(name, args, ctx, record)
            _save_step(ctx, call, step, record, output)
        record["output_preview"] = json.dumps(output, default=str)[:500]  # what the MODEL saw back
        return output, record
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def _decide(
    name: str, raw_args: str, ctx: ToolContext, authorize_fn: AuthorizeFn, approve_fn: ApproveFn, record: dict
):
    """validate -> authorize -> (approve). Returns (args to execute, None) or (None, output for the model)."""
    try:
        args = validate_call(name, raw_args)
    except ToolError as e:
        record["status"] = "invalid"
        record["error"] = str(e)
        return None, {"error": str(e)}

    record["validated_args"] = args.model_dump()  # what actually runs (raw text can mislead, e.g. duplicate keys)
    tool_call = ToolCall(name, args)
    decision = authorize_fn(tool_call)
    record["decision"] = decision
    if decision == "allow":
        return args, None
    if decision == "needs_approval":
        approved = approve_fn(tool_call, ctx) is True  # only a literal True counts
        record["approved"] = approved
        if approved:
            return args, None
        record["status"] = "rejected"
        message = (
            f"DECLINED: the user said no to {name}. It was NOT done. "
            "Tell the user plainly that they declined it. Do not retry unless they ask."
        )
        return None, {"error": message + _already_sent_note(ctx, args)}
    record["status"] = "denied"  # "deny" -- or any unexpected value: fail closed
    return None, {"error": f"{name} is not permitted"}


def _save_approval(ctx: ToolContext, record: dict) -> None:
    """If a human was asked about a draft, record their answer NOW, in its own transaction."""
    draft_id = (record.get("validated_args") or {}).get("draft_id")
    if "approved" in record and draft_id:
        with ctx.conn.transaction():
            state.record_approval(ctx.conn, draft_id, record["approved"], ctx.run_id)  # no-op if no such draft


def _save_step(ctx: ToolContext, call: dict, step: int, record: dict, output: dict) -> None:
    """Inside the step transaction: the step row."""
    if ctx.run_id is None:
        return
    state.add_step(
        ctx.conn,
        ctx.run_id,
        turn=step,
        kind="tool_call",
        call_id=call["call_id"],
        tool=call["name"],
        payload={"arguments": call.get("arguments", ""), "validated_args": record.get("validated_args")},
        decision=_enforced(record.get("decision")),
        approved=record.get("approved"),
        status=record.get("status"),
        output=output,
    )


def _enforced(decision) -> str | None:
    """What the gate actually DID. An unexpected value from authorize_fn was enforced as deny, so
    store "deny" (the JSONL log keeps the raw value for debugging)."""
    if decision is None or decision in ("allow", "deny", "needs_approval"):
        return decision
    return "deny"


def run_agent(
    task: str,
    ctx: ToolContext,
    *,
    max_steps: int = 6,
    respond_fn: RespondFn | None = None,
    authorize_fn: AuthorizeFn = authorize,
    approve_fn: ApproveFn = no_approver,
    settings: Settings | None = None,
    crash_after: str | None = None,
) -> AgentResult:
    """Loop until the model answers in text, or raise MaxStepsError. No approver -> needs_approval = no.

    Persists the run in Postgres (runs + run_steps). Give it an AUTOCOMMIT connection so each
    `with conn.transaction()` really commits; on a non-autocommit connection with a transaction
    already open, every block is only a savepoint and nothing commits until the caller does
    (that's how tests and evals keep the DB clean).
    """
    settings = settings or get_settings()
    with ctx.conn.transaction():
        ctx.run_id = state.start_run(ctx.conn, task, settings.model, AGENT_PROMPT_VERSION)
    loop = _Loop(ctx, max_steps, _respond_fn(respond_fn, settings), authorize_fn, approve_fn, settings, crash_after)
    return loop.drive([{"role": "user", "content": task}], first_turn=1)


def resume_agent(
    run_id: UUID,
    ctx: ToolContext,
    *,
    max_steps: int = 6,
    respond_fn: RespondFn | None = None,
    authorize_fn: AuthorizeFn = authorize,
    approve_fn: ApproveFn = no_approver,
    settings: Settings | None = None,
    crash_after: str | None = None,
) -> AgentResult:
    """Continue a crashed/failed run after its last committed step. Committed steps are never re-run:
    their outputs are replayed from run_steps. max_steps counts ALL turns, old and new."""
    settings = settings or get_settings()
    run = state.get_run(ctx.conn, run_id)
    if run is None:
        raise ResumeError(f"no run {run_id}")
    if run["status"] == "done":
        raise ResumeError(f"run {run_id} already finished")
    if (run["agent_prompt_version"], run["model"]) != (AGENT_PROMPT_VERSION, settings.model):
        raise ResumeError(
            f"run {run_id} was started with {run['agent_prompt_version']} / {run['model']}, now "
            f"{AGENT_PROMPT_VERSION} / {settings.model}: the conversation can't continue under different rules"
        )
    steps = state.get_steps(ctx.conn, run_id)
    items, last_turn, pending, answer = rebuild(run["task"], steps)
    ctx.run_id = run_id
    ctx.warnings = warnings_from_steps(steps)
    with ctx.conn.transaction():
        state.mark_running(ctx.conn, run_id)
    loop = _Loop(ctx, max_steps, _respond_fn(respond_fn, settings), authorize_fn, approve_fn, settings, crash_after)
    if answer is not None:  # crashed after the final answer was saved, before finish_run
        return loop.finish(AgentResult(answer=answer, steps=last_turn, tool_calls=[], run_id=run_id))
    return loop.drive(items, first_turn=last_turn + 1, pending=pending)


def rebuild(task: str, steps: list[dict]) -> tuple[list[dict], int, list[dict], str | None]:
    """Stored steps -> (model input so far, last turn, calls still to run, final answer or None).

    model_turn -> its output items, replayed exactly as stored (shape varies: not always reasoning first).
    tool_call  -> the function_call_output the model was sent.
    pending    -> calls in the LAST model turn with no committed tool step (e.g. 2 calls, 1 committed).
    """
    items: list[dict] = [{"role": "user", "content": task}]
    done = {s["call_id"] for s in steps if s["kind"] == "tool_call"}
    last_turn, last_body = 0, None
    for s in steps:
        if s["kind"] == "model_turn":
            items += s["payload"]
            last_turn, last_body = s["turn"], {"status": "completed", "output": s["payload"]}
        else:
            items.append(function_call_output(s["call_id"], json.dumps(s["output"], default=str)))
    if last_body is None:
        return items, 0, [], None
    calls = extract_function_calls(last_body)
    if not calls:
        return items, last_turn, [], extract_output_text(last_body)
    return items, last_turn, [c for c in calls if c["call_id"] not in done], None


def warnings_from_steps(steps: list[dict]) -> list[str]:
    """Approval warnings from stored search results, so they survive a crash."""
    out = []
    for s in steps:
        if s["kind"] == "tool_call" and s["tool"] == "search_company" and s["status"] == "ok":
            question = (s["payload"].get("validated_args") or {}).get("question", "")
            if warning := search_warning(question, s["output"] or {}):
                out.append(warning)
    return out


def _respond_fn(respond_fn: RespondFn | None, settings: Settings) -> RespondFn:
    if respond_fn is not None:
        return respond_fn
    client = make_client(settings)
    return lambda items, tools: respond(items, tools, client=client, settings=settings)


@dataclass
class _Loop:
    """The agent loop, shared by run_agent and resume_agent."""

    ctx: ToolContext
    max_steps: int
    respond_fn: RespondFn
    authorize_fn: AuthorizeFn
    approve_fn: ApproveFn
    settings: Settings
    crash_after: str | None = None

    def drive(self, items: list[dict], first_turn: int, pending: list[dict] | None = None) -> AgentResult:
        try:
            result = self._loop(items, first_turn, pending or [])
        except Exception:  # NOT SimulatedCrash / KeyboardInterrupt: those leave the run 'running', like a kill
            with self.ctx.conn.transaction():
                state.finish_run(self.ctx.conn, self.ctx.run_id, "failed")
            raise
        return self.finish(result)

    def finish(self, result: AgentResult) -> AgentResult:
        with self.ctx.conn.transaction():
            state.finish_run(self.ctx.conn, self.ctx.run_id, "done", result.answer)
        return result

    def _loop(self, items: list[dict], first_turn: int, pending: list[dict]) -> AgentResult:
        tools = tool_definitions()
        records: list[dict] = []
        if pending:  # resume: finish the interrupted turn first, then ask the model again
            self._run_calls(pending, first_turn - 1, items, records)
        for step in range(first_turn, self.max_steps + 1):
            body = self.respond_fn(items, tools)
            with self.ctx.conn.transaction():
                state.add_step(self.ctx.conn, self.ctx.run_id, turn=step, kind="model_turn", payload=body["output"])
            items += body["output"]  # reasoning + calls go back, in order
            calls = extract_function_calls(body)
            if not calls:
                return AgentResult(
                    answer=extract_output_text(body), steps=step, tool_calls=records, run_id=self.ctx.run_id
                )
            self._run_calls(calls, step, items, records)
        raise MaxStepsError(f"no final answer after {self.max_steps} steps ({len(records)} tool calls)")

    def _run_calls(self, calls: list[dict], step: int, items: list[dict], records: list[dict]) -> None:
        for call in calls:  # EVERY call needs an output, or the next request is a 400
            output, record = handle_call(call, self.ctx, self.authorize_fn, step, self.settings, self.approve_fn)
            records.append(record)
            items.append(function_call_output(call["call_id"], json.dumps(output, default=str)))
            if call["name"] == self.crash_after:
                raise SimulatedCrash(f"simulated crash after {call['name']} (its step is committed)")


def _already_sent_note(ctx: ToolContext, args) -> str:
    """A decline only means THIS attempt didn't happen. If the ledger says the draft went out
    earlier, the model must know, or it tells the user "it was not sent" (seen live in M5)."""
    draft_id = getattr(args, "draft_id", None)
    send = state.get_send(ctx.conn, draft_id) if draft_id else None
    if send is None or send["status"] != "sent":
        return ""
    return (
        f" Note: draft {draft_id} was ALREADY SENT earlier, at {send['sent_at']:%Y-%m-%d %H:%M}; "
        "that email is unaffected. Do not tell the user it was never sent."
    )


def _fit_warnings(ctx: ToolContext, draft) -> list[str]:
    """D5 decision: a lead drafted on a POSSIBLE fit is flagged with its unknown criteria."""
    if draft is None:
        return []
    warning = fit_warning(state.latest_qualification(ctx.conn, draft.lead_id))
    return [warning] if warning else []


def _ledger_warnings(ctx: ToolContext, draft) -> list[str]:
    """What the sends ledger says about this draft: approving must never look like it will send
    something that won't be sent (or might already have been)."""
    send = state.get_send(ctx.conn, draft.draft_id) if draft is not None else None
    if send is None:
        return []
    if send["status"] == "sent":
        return [f"ALREADY SENT at {send['sent_at']:%Y-%m-%d %H:%M}. Approving will NOT send it again."]
    return [
        (
            "An earlier send of this draft did not finish. Approving checks whether it went out "
            "and only sends if it did not."
        )
    ]


def cli_approve(call: ToolCall, ctx: ToolContext, input_fn: Callable[[str], str] = input) -> bool:
    """Show what will REALLY happen (from our own store, not the model's words) and ask y/N."""
    print(f"\n=== APPROVAL NEEDED: {call.name} ===", file=sys.stderr)
    draft = state.get_draft(ctx.conn, call.args.draft_id) if call.name == "send_email" and call.args else None
    for w in [*_ledger_warnings(ctx, draft), *_fit_warnings(ctx, draft), *ctx.warnings]:
        print(f"⚠ {w}", file=sys.stderr)
    if draft is not None:
        print(f"To: {draft.to}\nSubject: {draft.subject}\n\n{draft.body}\n", file=sys.stderr)
    else:
        print(f"args: {call.args.model_dump_json() if call.args else '(none)'}", file=sys.stderr)
    try:
        answer = input_fn("Approve? [y/N] ")
    except EOFError:  # no keyboard (piped input, Ctrl-D): treat as no
        return False
    return answer.strip().lower() in {"y", "yes"}


def main() -> None:
    from research_assistant.db import connect

    parser = argparse.ArgumentParser(prog="agent", description="Sales research agent with tools.")
    parser.add_argument("task", nargs="?", help='e.g. "Look up lead L2 and draft them a short intro email."')
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a crashed run after its last committed step")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--crash-after", metavar="TOOL", help=argparse.SUPPRESS)  # M4 demo: die after TOOL's step
    args = parser.parse_args()
    if bool(args.task) == bool(args.resume):
        parser.error('give a task, or --resume RUN_ID (not both)')

    with connect(autocommit=True) as conn:  # autocommit: each step's transaction really commits
        ctx = ToolContext(conn=conn)
        kw = {"max_steps": args.max_steps, "approve_fn": cli_approve, "crash_after": args.crash_after}
        try:
            if args.resume:
                result = resume_agent(UUID(args.resume), ctx, **kw)
            else:
                result = run_agent(args.task, ctx, **kw)
        except SimulatedCrash as e:
            print(f"\n💥 {e}. Run left as 'running'. Resume with:\n  uv run agent --resume {ctx.run_id}", file=sys.stderr)
            sys.exit(1)
        except (MaxStepsError, IncompleteResponseError, ResumeError) as e:
            print(f"error: {e} (run {ctx.run_id or args.resume})", file=sys.stderr)
            sys.exit(1)

        print(result.answer)
        print(f"\n({result.steps} steps, run {result.run_id})", file=sys.stderr)
        for r in result.tool_calls:
            approved = f" approved={r['approved']}" if "approved" in r else ""
            print(f"  step {r['step']}: {r['tool']} -> {r.get('decision', '-')}{approved} / {r['status']}", file=sys.stderr)
        for d in state.list_drafts(conn, result.run_id):
            sent = (ctx.outbox_dir / f"{d.draft_id}.eml").exists()
            label = f"SENT -> {ctx.outbox_dir / (d.draft_id + '.eml')}" if sent else "NOT sent"
            print(f"\n--- draft {d.draft_id} ({label}) to {d.to}\nSubject: {d.subject}\n\n{d.body}", file=sys.stderr)


if __name__ == "__main__":
    main()
