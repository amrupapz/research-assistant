"""Agent loop: model proposes -> validate -> authorize -> execute -> feed the result back. `agent "task"`."""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

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
from research_assistant.tools import (
    ToolContext,
    ToolError,
    execute,
    tool_definitions,
    validate_call,
)

RespondFn = Callable[[list[dict], list[dict]], dict]
AuthorizeFn = Callable[[ToolCall], Decision]
ApproveFn = Callable[[ToolCall, ToolContext], bool]


class MaxStepsError(Exception):
    """The model was still calling tools after max_steps turns."""


@dataclass
class AgentResult:
    answer: str
    steps: int  # model turns used
    tool_calls: list[dict]  # one log record per proposed call, in order


def no_approver(call: ToolCall, ctx: ToolContext) -> bool:
    """Default when nobody is at the keyboard (tests, evals, batch): needs_approval means no."""
    return False


def _execute(name: str, args, ctx: ToolContext, record: dict) -> dict:
    """Run an authorized (and, if needed, approved) call; turn failures into {"error": ...}."""
    try:
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
        try:
            args = validate_call(name, call.get("arguments", ""))
        except ToolError as e:
            record["status"] = "invalid"
            record["error"] = str(e)
            return {"error": str(e)}, record

        tool_call = ToolCall(name, args)
        decision = authorize_fn(tool_call)
        record["decision"] = decision
        if decision == "allow":
            return _execute(name, args, ctx, record), record
        if decision == "needs_approval":
            approved = approve_fn(tool_call, ctx) is True  # only a literal True counts
            record["approved"] = approved
            if approved:
                return _execute(name, args, ctx, record), record
            record["status"] = "rejected"
            return {"error": f"the user did not approve {name}; do not retry it unless they ask"}, record
        record["status"] = "denied"  # "deny" -- or any unexpected value: fail closed
        return {"error": f"{name} is not permitted"}, record
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def run_agent(
    task: str,
    ctx: ToolContext,
    *,
    max_steps: int = 6,
    respond_fn: RespondFn | None = None,
    authorize_fn: AuthorizeFn = authorize,
    approve_fn: ApproveFn = no_approver,
    settings: Settings | None = None,
) -> AgentResult:
    """Loop until the model answers in text, or raise MaxStepsError. No approver -> needs_approval = no."""
    settings = settings or get_settings()
    if respond_fn is None:
        client = make_client(settings)
        respond_fn = lambda items, tools: respond(items, tools, client=client, settings=settings)

    tools = tool_definitions()
    items: list[dict] = [{"role": "user", "content": task}]
    records: list[dict] = []
    for step in range(1, max_steps + 1):
        body = respond_fn(items, tools)
        items += body["output"]  # reasoning + calls go back, in order
        calls = extract_function_calls(body)
        if not calls:
            return AgentResult(answer=extract_output_text(body), steps=step, tool_calls=records)
        for call in calls:  # EVERY call needs an output, or the next request is a 400
            output, record = handle_call(call, ctx, authorize_fn, step, settings, approve_fn)
            records.append(record)
            items.append(function_call_output(call["call_id"], json.dumps(output, default=str)))
    raise MaxStepsError(f"no final answer after {max_steps} steps ({len(records)} tool calls)")


def cli_approve(call: ToolCall, ctx: ToolContext, input_fn: Callable[[str], str] = input) -> bool:
    """Show what will REALLY happen (from our own store, not the model's words) and ask y/N."""
    print(f"\n=== APPROVAL NEEDED: {call.name} ===", file=sys.stderr)
    draft = ctx.drafts.get(call.args.draft_id) if call.name == "send_email" and call.args else None
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
    parser.add_argument("task", help='e.g. "Look up lead L2 and draft them a short intro email."')
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    with connect() as conn:
        ctx = ToolContext(conn=conn)
        try:
            result = run_agent(args.task, ctx, max_steps=args.max_steps, approve_fn=cli_approve)
        except (MaxStepsError, IncompleteResponseError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

    print(result.answer)
    print(f"\n({result.steps} steps)", file=sys.stderr)
    for r in result.tool_calls:
        approved = f" approved={r['approved']}" if "approved" in r else ""
        print(f"  step {r['step']}: {r['tool']} -> {r.get('decision', '-')}{approved} / {r['status']}", file=sys.stderr)
    for d in ctx.drafts.values():
        sent = (ctx.outbox_dir / f"{d.draft_id}.eml").exists()
        label = f"SENT -> {ctx.outbox_dir / (d.draft_id + '.eml')}" if sent else "NOT sent"
        print(f"\n--- draft {d.draft_id} ({label}) to {d.to}\nSubject: {d.subject}\n\n{d.body}", file=sys.stderr)


if __name__ == "__main__":
    main()
