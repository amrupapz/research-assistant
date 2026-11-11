"""The sales flow as a LangGraph workflow (D5). CODE decides the order; the model works inside nodes.

    START → load_lead ─┬─ replied → END
                       └─ research → qualify ──route_after_qualify──┬─ "stop"     → stop → END
                             ↑                                      ├─ "research" → research (extra pass)
                             └──────────────────────────────────────┘
                                                                    └─ "draft"    → draft ─┐
                                                       ┌─ yes → send → END                       │
                                    approve ⏸ ─────────┤                    ←────────────────────┘
                                                       └─ no  → END (declined)

Every tool still goes through validate → authorize → execute (call_tool). send_email is the one
needs_approval tool: only the send node runs it, only after approve recorded a "yes" in MY approvals
table, and through send_once (so a re-run of the send node can't send twice).

    uv run python -m research_assistant.flow L1
    uv run python -m research_assistant.flow --resume <run_id>   (after a crash)
    uv run agent approve D20 y|n                                  (resume a run waiting at approve)

M2: a Postgres checkpointer saves the state after every node, keyed by thread_id = run_id. Resume =
invoke(None) on that thread: finished nodes are skipped, the node that crashed runs again FROM ITS START.
M3: approve calls interrupt(): the run is saved and the process exits ('waiting'). `agent approve`
resumes it with Command(resume=True|False); approve re-runs from its start and interrupt() returns that.
"""

import argparse
import json
import sys
from typing import Literal, TypedDict
from uuid import UUID

import psycopg
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row

from research_assistant import state
from research_assistant.config import Settings, get_settings
from research_assistant.llm import write_email
from research_assistant.permissions import ToolCall, authorize
from research_assistant.qualify import fit_warning, load_icp
from research_assistant.tools import ToolContext, ToolError, execute, validate_call

FLOW_VERSION = "flow-v2"  # stored in runs.agent_prompt_version, so `agent --resume` refuses flow runs
MAX_RESEARCH_PASSES = 2  # the first pass + exactly one extra pass for "possible"
DEEP_QUALIFY_K = 8  # chunks per criterion on the extra pass (the normal pass uses 4)


class FlowState(TypedDict, total=False):
    lead_id: str
    lead: dict
    outreach: dict  # from lead_status: what we already sent, replies
    findings: list[dict]  # one entry per search_company call: {"question", answer fields...} or {"question", "error"}
    research_passes: int  # research passes COMPLETED so far (1 after the first pass)
    qualification: dict  # qualify_lead output: fit, criteria, icp_version
    draft: dict  # draft_email output
    warnings: list[str]  # shown at approval time, e.g. "possible fit: scale=unknown"
    approved: bool  # the human's decision at approve (also in MY approvals table)
    sent: dict  # send_email output
    outcome: str  # how the run ended, in words


class FlowError(Exception):
    """A tool the workflow needs was not allowed to run, or a run can't be resumed."""


Route = Literal["research", "draft", "stop"]


def route_after_qualify(state: FlowState) -> Route:
    """Decide where the flow goes after qualify. Reads state; changes nothing; does no work.

    ✍️ YOU WRITE THIS (Project 5 core function).

    Contract (your D5 decision):
        fit       after pass 1 (research_passes == 1)   after the extra pass (research_passes >= 2)
        poor      "stop"                                "stop"
        strong    "draft"                               "draft"
        possible  "research"  (exactly one extra pass)  "draft"  (the draft node flags it ⚠)

      - fit is state["qualification"]["fit"]; the pass count is state["research_passes"].
      - Return only "research", "draft" or "stop": they are node names in build_graph.
      - Anything unexpected (no qualification, an unknown fit value) → "stop". Fail closed:
        never draft to a lead you couldn't qualify.

    Hints:
      - Use MAX_RESEARCH_PASSES (2) rather than a bare 2, so "one extra pass" lives in one place.
      - .get() with a default for anything that might be missing.
      - ~8 lines. Run: uv run pytest tests/test_flow.py -k route -v
    """
    qualification = state.get("qualification") or {}
    fit = qualification.get("fit")
    passes = state.get("research_passes", MAX_RESEARCH_PASSES)

    if fit == "strong":
        return "draft"
    if fit == "possible":
        return "research" if passes < MAX_RESEARCH_PASSES else "draft"
    return "stop"


def call_tool(ctx: ToolContext, name: str, args: dict) -> dict:
    """The P3 gate for code-driven calls: validate → authorize → execute (one transaction).

    Only "allow" runs here. needs_approval (send_email) is NOT handled by this helper: that's the
    approval node's job (M3), so the workflow can never send by accident.
    """
    parsed = validate_call(name, json.dumps(args))
    decision = authorize(ToolCall(name, parsed))
    if decision != "allow":
        raise FlowError(f"{name} is not allowed here (decision={decision})")
    with ctx.conn.transaction():
        return execute(name, parsed, ctx)


# --- nodes: each takes the state, returns ONLY the fields it changes -----------------------


def checkpointer_conn(url: str) -> psycopg.Connection:
    """The checkpointer's OWN connection: autocommit (each checkpoint commits on its own, so it
    survives a crash) and dict rows (PostgresSaver reads columns by name). Not ctx.conn."""
    return psycopg.connect(url, autocommit=True, row_factory=dict_row)


def setup_checkpointer(url: str) -> None:
    """Create/upgrade LangGraph's checkpoint tables. They're the library's, not Alembic's: it keeps
    its own version table (checkpoint_migrations). Safe to run every time."""
    from langgraph.checkpoint.postgres import PostgresSaver

    with checkpointer_conn(url) as conn:
        PostgresSaver(conn).setup()


def build_graph(ctx: ToolContext, checkpointer=None):
    """Compile the workflow. ctx (DB connection, outbox, ...) is closed over, never put in state:
    state must stay plain data so the checkpointer (M2) can save it."""
    from langgraph.graph import END, START, StateGraph

    def load_lead(s: FlowState) -> FlowState:
        status = call_tool(ctx, "lead_status", {"lead_id": s["lead_id"]})
        update: FlowState = {"lead": status["lead"], "outreach": status["outreach"], "findings": [], "research_passes": 0}
        if status["outreach"]["replied_since_last_contact"]:
            update["outcome"] = "stopped: the lead replied since our last email"
        return update

    def after_load(s: FlowState) -> Literal["research", "__end__"]:
        return END if s.get("outcome") else "research"

    def research(s: FlowState) -> FlowState:
        company = s["lead"]["company"]
        passes = s.get("research_passes", 0) + 1
        if passes == 1:
            questions = [f"What does {company} do?"]
        else:  # the extra pass: ask about exactly the criteria qualify couldn't decide
            unknown = {c["id"] for c in s["qualification"]["criteria"] if c["met"] == "unknown"}
            icp = load_icp(ctx.icp_path)
            questions = [c.question_for(company) for c in icp.evidence_criteria() if c.id in unknown]
        findings = list(s.get("findings", []))
        for q in questions:
            try:
                findings.append({"question": q, **call_tool(ctx, "search_company", {"company": company, "question": q})})
            except ToolError as e:
                findings.append({"question": q, "error": str(e)})
        return {"findings": findings, "research_passes": passes}

    def qualify(s: FlowState) -> FlowState:
        if s.get("research_passes", 0) > 1:  # extra pass: look deeper, and don't reuse the old verdict
            ctx.qualify_k, ctx.reuse_qualification = DEEP_QUALIFY_K, False
        try:
            return {"qualification": call_tool(ctx, "qualify_lead", {"lead_id": s["lead_id"]})}
        finally:
            ctx.qualify_k, ctx.reuse_qualification = 4, True

    def stop(s: FlowState) -> FlowState:
        fit = (s.get("qualification") or {}).get("fit", "unknown")
        return {"outcome": f"stopped: {fit} fit"}

    def draft(s: FlowState) -> FlowState:
        email = write_email(s["lead"], s["outreach"]["sent"], s.get("findings", []))
        try:
            out = call_tool(
                ctx, "draft_email", {"lead_id": s["lead_id"], "subject": email.subject, "body": email.body}
            )
        except ToolError as e:  # e.g. the model added a link: rejected by draft_email's validation
            return {"outcome": f"error: draft rejected ({e})"}
        warnings = [w for w in [fit_warning(s.get("qualification"))] if w]
        return {"draft": out, "warnings": warnings, "outcome": f"drafted {out['draft_id']} (awaiting approval)"}

    def after_draft(s: FlowState) -> Literal["approve", "__end__"]:
        return "approve" if s.get("draft") else END  # a rejected draft has nothing to approve

    def approve(s: FlowState) -> FlowState:
        # On resume this node RE-RUNS FROM HERE: nothing with a side effect may come before interrupt().
        d = s["draft"]
        answer = interrupt({"draft_id": d["draft_id"], "to": d["to"], "warnings": s.get("warnings", [])})
        approved = answer is True  # only a literal True counts, like cli_approve
        with ctx.conn.transaction():  # the decision is a business fact: MY table (send_once checks it there)
            state.record_approval(ctx.conn, d["draft_id"], approved, ctx.run_id)
        update: FlowState = {"approved": approved}
        if not approved:
            update["outcome"] = f"declined: {d['draft_id']} not sent"
        return update

    def after_approve(s: FlowState) -> Literal["send", "__end__"]:
        return "send" if s.get("approved") is True else END

    def send(s: FlowState) -> FlowState:
        d = s["draft"]
        parsed = validate_call("send_email", json.dumps({"draft_id": d["draft_id"]}))
        decision = authorize(ToolCall("send_email", parsed))
        if decision != "needs_approval":  # deny → never; "allow" would mean the rules changed under us
            raise FlowError(f"send_email is not allowed here (decision={decision})")
        try:  # NOT inside a transaction: send_once commits its own claim/send steps (D4 M5)
            out = execute("send_email", parsed, ctx)
        except ToolError as e:
            return {"outcome": f"error: send failed ({e})"}
        return {"sent": out, "outcome": f"{out['status']}: {d['draft_id']} to {d['to']}"}

    g = StateGraph(FlowState)
    nodes = [("load_lead", load_lead), ("research", research), ("qualify", qualify), ("stop", stop),
             ("draft", draft), ("approve", approve), ("send", send)]  # fmt: skip
    for name, fn in nodes:
        g.add_node(name, fn)
    g.add_edge(START, "load_lead")
    g.add_conditional_edges("load_lead", after_load, ["research", END])
    g.add_edge("research", "qualify")
    g.add_conditional_edges("qualify", route_after_qualify, ["research", "draft", "stop"])
    g.add_edge("stop", END)
    g.add_conditional_edges("draft", after_draft, ["approve", END])
    g.add_conditional_edges("approve", after_approve, ["send", END])
    g.add_edge("send", END)
    return g.compile(checkpointer=checkpointer)


def _config(run_id: UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


def _invoke(ctx: ToolContext, checkpointer, graph_input) -> FlowState:
    """Run (or continue) the graph on thread run_id; keep the runs row's status in step:
    paused at interrupt() → 'waiting', reached END → 'done', raised → 'failed'."""
    graph = build_graph(ctx, checkpointer)
    try:
        final = graph.invoke(graph_input, _config(ctx.run_id))
    except BaseException:  # BaseException: Ctrl+C (KeyboardInterrupt) is NOT an Exception
        with ctx.conn.transaction():
            state.finish_run(ctx.conn, ctx.run_id, "failed")
        raise
    paused = bool(graph.get_state(_config(ctx.run_id)).next)  # nodes still to run = stopped at interrupt()
    with ctx.conn.transaction():
        if paused:
            state.wait_run(ctx.conn, ctx.run_id, final.get("outcome"))
        else:
            state.finish_run(ctx.conn, ctx.run_id, "done", final.get("outcome"))
    return final


def run_flow(lead_id: str, ctx: ToolContext, settings: Settings | None = None, checkpointer=None) -> FlowState:
    """Run the workflow for one lead, recorded as a `runs` row; thread_id = run_id.

    No checkpointer → an in-memory one (tests): interrupt() needs SOME checkpointer to pause."""
    from langgraph.checkpoint.memory import InMemorySaver

    settings = settings or get_settings()
    checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    with ctx.conn.transaction():
        ctx.run_id = state.start_run(ctx.conn, f"flow {lead_id}", settings.model, FLOW_VERSION)
    return _invoke(ctx, checkpointer, {"lead_id": lead_id})


def _flow_run(ctx: ToolContext, run_id: UUID) -> dict:
    run = state.get_run(ctx.conn, run_id)
    if run is None or run["agent_prompt_version"] != FLOW_VERSION:
        made_by = f" (made by {run['agent_prompt_version']})" if run else ""
        raise FlowError(f"run {run_id} is not a {FLOW_VERSION} run{made_by}")
    return run


def resume_flow(run_id: UUID, ctx: ToolContext, checkpointer) -> FlowState:
    """Continue a crashed flow run from its last checkpoint. Refuses runs that aren't resumable."""
    run = _flow_run(ctx, run_id)
    if run["status"] == "done":
        raise FlowError(f"run {run_id} already finished")
    if run["status"] == "waiting":
        raise FlowError(f"run {run_id} is waiting for approval: uv run agent approve <draft_id> y|n")
    if checkpointer.get_tuple(_config(run_id)) is None:
        raise FlowError(f"run {run_id} has no checkpoint to resume from")
    ctx.run_id = run_id
    with ctx.conn.transaction():
        state.mark_running(ctx.conn, run_id)
    return _invoke(ctx, checkpointer, None)  # None = "no new input: continue from the checkpoint"


def approve_flow(draft_id: str, approved: bool, ctx: ToolContext, checkpointer) -> FlowState:
    """Resume the run that is waiting at approve for THIS draft, with the human's decision."""
    run_id = state.draft_run_id(ctx.conn, draft_id)
    if run_id is None:
        raise FlowError(f"no draft {draft_id} from a workflow run")
    run = _flow_run(ctx, run_id)
    snapshot = build_graph(ctx, checkpointer).get_state(_config(run_id))
    waiting_for = (snapshot.values.get("draft") or {}).get("draft_id") if snapshot.next == ("approve",) else None
    if run["status"] != "waiting" or waiting_for != draft_id:
        raise FlowError(f"draft {draft_id} is not waiting for approval (run {run_id} is {run['status']})")
    ctx.run_id = run_id
    with ctx.conn.transaction():
        state.mark_running(ctx.conn, run_id)
    return _invoke(ctx, checkpointer, Command(resume=approved))


def main() -> None:
    from langgraph.checkpoint.postgres import PostgresSaver

    from research_assistant.db import connect

    parser = argparse.ArgumentParser(prog="flow", description="Run the sales workflow for one lead.")
    parser.add_argument("lead_id", nargs="?", help="e.g. L1")
    parser.add_argument("--resume", metavar="RUN_ID", type=UUID, help="continue a crashed flow run")
    args = parser.parse_args()
    if (args.lead_id is None) == (args.resume is None):
        parser.error("give a lead_id OR --resume RUN_ID")
    url = get_settings().database_url
    with connect(autocommit=True) as conn, checkpointer_conn(url) as cp_conn:
        ctx = ToolContext(conn=conn)
        saver = PostgresSaver(cp_conn)
        try:
            final = resume_flow(args.resume, ctx, saver) if args.resume else run_flow(args.lead_id, ctx, checkpointer=saver)
        except FlowError as e:
            sys.exit(f"flow: {e}")
        except BaseException:
            print(f"flow crashed; resume with: uv run python -m research_assistant.flow --resume {ctx.run_id}", file=sys.stderr)
            raise
    print(final.get("outcome"))
    print(f"(run {ctx.run_id}, research passes: {final.get('research_passes', 0)})", file=sys.stderr)
    q = final.get("qualification") or {}
    if q:
        print(f"fit: {q.get('fit')}  " + ", ".join(f"{c['id']}={c['met']}" for c in q.get("criteria", [])), file=sys.stderr)
    for w in final.get("warnings", []):
        print(f"⚠ {w}", file=sys.stderr)
    if (d := final.get("draft")) and "approved" not in final:
        print(f"\n--- draft {d['draft_id']} to {d['to']}\nSubject: {d['subject']}\n\n{d['body']}", file=sys.stderr)
        print(f"\n⏸ waiting for you. Approve or decline (now or tomorrow):\n  uv run agent approve {d['draft_id']} y|n", file=sys.stderr)


def approve_main(argv: list[str]) -> None:
    """`agent approve D20 y|n`: resume the paused workflow for that draft."""
    from langgraph.checkpoint.postgres import PostgresSaver

    from research_assistant.db import connect

    parser = argparse.ArgumentParser(prog="agent approve", description="Approve or decline a waiting draft.")
    parser.add_argument("draft_id", help="e.g. D20")
    parser.add_argument("decision", choices=["y", "n"])
    args = parser.parse_args(argv)
    url = get_settings().database_url
    with connect(autocommit=True) as conn, checkpointer_conn(url) as cp_conn:
        ctx = ToolContext(conn=conn)
        try:
            final = approve_flow(args.draft_id, args.decision == "y", ctx, PostgresSaver(cp_conn))
        except FlowError as e:
            sys.exit(f"approve: {e}")
        except BaseException:
            print(f"crashed; resume with: uv run python -m research_assistant.flow --resume {ctx.run_id}", file=sys.stderr)
            raise
    print(final.get("outcome"))
    print(f"(run {ctx.run_id})", file=sys.stderr)


if __name__ == "__main__":
    main()
