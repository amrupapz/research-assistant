"""The sales flow as a LangGraph workflow (D5). CODE decides the order; the model works inside nodes.

    START → load_lead ─┬─ replied → END
                       └─ research → qualify ──route_after_qualify──┬─ "stop"     → stop → END
                             ↑                                      ├─ "research" → research (extra pass)
                             └──────────────────────────────────────┘
                                                                    └─ "draft"    → draft → END
                                                                                    (M3: approve ⏸ → send)

Every tool still goes through validate → authorize → execute (call_tool). The graph itself never
sends: there is no send node until M3/M4, and send_email needs approval anyway.

    uv run python -m research_assistant.flow L1
    uv run python -m research_assistant.flow --resume <run_id>   (after a crash)

M2: a Postgres checkpointer saves the state after every node, keyed by thread_id = run_id. Resume =
invoke(None) on that thread: finished nodes are skipped, the node that crashed runs again FROM ITS START.
"""

import argparse
import json
import sys
from typing import Literal, TypedDict
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from research_assistant import state
from research_assistant.config import Settings, get_settings
from research_assistant.llm import write_email
from research_assistant.permissions import ToolCall, authorize
from research_assistant.qualify import fit_warning, load_icp
from research_assistant.tools import ToolContext, ToolError, execute, validate_call

FLOW_VERSION = "flow-v1"  # stored in runs.agent_prompt_version, so `agent --resume` refuses flow runs
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

    g = StateGraph(FlowState)
    for name, fn in [("load_lead", load_lead), ("research", research), ("qualify", qualify), ("stop", stop), ("draft", draft)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_lead")
    g.add_conditional_edges("load_lead", after_load, ["research", END])
    g.add_edge("research", "qualify")
    g.add_conditional_edges("qualify", route_after_qualify, ["research", "draft", "stop"])
    g.add_edge("stop", END)
    g.add_edge("draft", END)
    return g.compile(checkpointer=checkpointer)


def _config(run_id: UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


def _invoke(ctx: ToolContext, checkpointer, graph_input: FlowState | None) -> FlowState:
    """Run (or continue) the graph on thread run_id; keep the runs row's status in step."""
    try:
        final = build_graph(ctx, checkpointer).invoke(graph_input, _config(ctx.run_id))
    except BaseException:  # BaseException: Ctrl+C (KeyboardInterrupt) is NOT an Exception
        with ctx.conn.transaction():
            state.finish_run(ctx.conn, ctx.run_id, "failed")
        raise
    with ctx.conn.transaction():
        state.finish_run(ctx.conn, ctx.run_id, "done", final.get("outcome"))
    return final


def run_flow(lead_id: str, ctx: ToolContext, settings: Settings | None = None, checkpointer=None) -> FlowState:
    """Run the workflow for one lead, recorded as a `runs` row; thread_id = run_id."""
    settings = settings or get_settings()
    with ctx.conn.transaction():
        ctx.run_id = state.start_run(ctx.conn, f"flow {lead_id}", settings.model, FLOW_VERSION)
    return _invoke(ctx, checkpointer, {"lead_id": lead_id})


def resume_flow(run_id: UUID, ctx: ToolContext, checkpointer) -> FlowState:
    """Continue a crashed flow run from its last checkpoint. Refuses runs that aren't resumable."""
    run = state.get_run(ctx.conn, run_id)
    if run is None or run["agent_prompt_version"] != FLOW_VERSION:
        raise FlowError(f"run {run_id} is not a flow run")
    if run["status"] == "done":
        raise FlowError(f"run {run_id} already finished")
    if checkpointer.get_tuple(_config(run_id)) is None:
        raise FlowError(f"run {run_id} has no checkpoint to resume from")
    ctx.run_id = run_id
    with ctx.conn.transaction():
        state.mark_running(ctx.conn, run_id)
    return _invoke(ctx, checkpointer, None)  # None = "no new input: continue from the checkpoint"


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
    if d := final.get("draft"):
        print(f"\n--- draft {d['draft_id']} to {d['to']}\nSubject: {d['subject']}\n\n{d['body']}", file=sys.stderr)


if __name__ == "__main__":
    main()
