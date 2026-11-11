"""D5 M1: the sales flow as a LangGraph workflow. Fake model + real test DB (rolled back).

The route_after_qualify tests are the spec for YOUR function. The graph tests further down run the
whole workflow, so they also fail until route_after_qualify is written.
"""

import json
from datetime import date

import pytest
from test_agent import _settings

from research_assistant import flow, state, tools
from research_assistant.agent import cli_approve
from research_assistant.flow import (
    MAX_RESEARCH_PASSES,
    FlowError,
    approve_flow,
    call_tool,
    resume_flow,
    route_after_qualify,
    run_flow,
)
from research_assistant.permissions import ToolCall
from research_assistant.qualify import fit_warning
from research_assistant.schemas import (
    Chunk,
    Citation,
    CriterionResult,
    EmailDraft,
    IcpJudgment,
    ResearchAnswer,
)
from research_assistant.sending import send_once
from research_assistant.tools import SendEmailArgs, ToolContext

# --- route_after_qualify: the six cells of the D5 decision table ----------------------------


def _s(fit: str | None, passes: int) -> dict:
    s = {"research_passes": passes}
    if fit is not None:
        s["qualification"] = {"fit": fit, "criteria": []}
    return s


@pytest.mark.parametrize(
    ("fit", "passes", "expected"),
    [
        ("poor", 1, "stop"),
        ("poor", 2, "stop"),
        ("strong", 1, "draft"),
        ("strong", 2, "draft"),
        ("possible", 1, "research"),  # exactly one extra pass...
        ("possible", 2, "draft"),  # ...then draft (flagged by the draft node)
    ],
    ids=["poor-1", "poor-2", "strong-1", "strong-2", "possible-1", "possible-2"],
)
def test_route_after_qualify_table(fit, passes, expected):
    assert route_after_qualify(_s(fit, passes)) == expected


def test_route_never_researches_more_than_the_limit():
    assert route_after_qualify(_s("possible", MAX_RESEARCH_PASSES + 3)) == "draft"


@pytest.mark.parametrize(
    "s",
    [{"research_passes": 1}, _s("great", 1), _s(None, 2), {}],
    ids=["no_qualification", "unknown_fit", "no_qualification_pass2", "empty_state"],
)
def test_route_fails_closed(s):
    assert (
        route_after_qualify(s) == "stop"
    )  # never draft to a lead you couldn't qualify


def test_route_does_not_change_the_state():
    s = _s("possible", 1)
    before = json.dumps(s, sort_keys=True)
    route_after_qualify(s)
    assert json.dumps(s, sort_keys=True) == before


# --- fit_warning: the ⚠ flag for possible fits -----------------------------------------------


def _q(fit, **met):
    return {"fit": fit, "criteria": [{"id": k, "met": v} for k, v in met.items()]}


def test_fit_warning_lists_unknown_criteria():
    q = _q(
        "possible",
        eng_buyer="yes",
        software_org="yes",
        scale="unknown",
        dev_tooling="unknown",
    )
    assert fit_warning(q) == "possible fit: scale=unknown, dev_tooling=unknown"


@pytest.mark.parametrize(
    "q", [_q("strong", scale="yes"), _q("poor", scale="unknown"), None, {}]
)
def test_no_fit_warning_unless_possible(q):
    assert fit_warning(q) is None


# --- the graph, end to end ------------------------------------------------------------------

CHUNK = Chunk(
    chunk_id="figma-about#000",
    source_url="https://figma.com/about",
    fetched_at=date(2026, 11, 1),
    title="About Figma",
    text="Figma helps teams design.",
)
ANSWER = ResearchAnswer(
    answer="Figma helps product teams design [figma-about#000]",
    confidence="high",
    assumptions=[],
    needs_sources=False,
    citations=[
        Citation(chunk_id="figma-about#000", source_url="https://figma.com/about")
    ],
)
STRONG = {"software_org": "yes", "scale": "yes", "dev_tooling": "yes"}
POSSIBLE = {"software_org": "yes", "scale": "unknown", "dev_tooling": "unknown"}
POOR = {"software_org": "no", "scale": "yes", "dev_tooling": "yes"}


@pytest.fixture
def ctx(state_db, tmp_path):
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


@pytest.fixture
def settings(tmp_path):
    return _settings(tmp_path / "runs")


@pytest.fixture
def fake(monkeypatch):
    """Fake retrieval, research answers, ICP judge and email writer. Records what was asked."""
    seen = {"search_k": [], "asked": [], "judged": 0, "emails": 0}
    verdicts: list[dict] = []

    def hybrid_search(conn, q, k):
        seen["search_k"].append(k)
        return [(CHUNK, 1.0)]

    def ask(q, chunks):
        seen["asked"].append(q)
        return ANSWER

    def judge(company, criteria, chunks):
        v = verdicts[min(seen["judged"], len(verdicts) - 1)]
        seen["judged"] += 1
        cite = lambda met: ["figma-about#000"] if met != "unknown" else []
        return IcpJudgment(
            results=[
                CriterionResult(id=cid, met=v[cid], reason="r", chunk_ids=cite(v[cid]))
                for cid, _ in criteria
            ]
        )

    def write(lead, sent, findings):
        seen["emails"] += 1
        return EmailDraft(
            subject="Hi", body="Hello Dana, Figma helps product teams design."
        )

    monkeypatch.setattr(tools, "hybrid_search", hybrid_search)
    monkeypatch.setattr(tools, "ask", ask)
    monkeypatch.setattr(tools, "judge_icp", judge)
    monkeypatch.setattr(flow, "write_email", write)
    seen["verdicts"] = verdicts
    return seen


def _drafts(ctx):
    return state.list_drafts(ctx.conn)


def test_strong_fit_drafts_after_one_pass(ctx, settings, fake):
    fake["verdicts"].append(STRONG)
    final = run_flow("L1", ctx, settings)
    assert (final["research_passes"], fake["judged"]) == (1, 1)
    assert final["qualification"]["fit"] == "strong"
    assert [d.draft_id for d in _drafts(ctx)] == ["D1"]
    assert (
        final["warnings"] == [] and final["outcome"] == "drafted D1 (awaiting approval)"
    )
    run = state.get_run(ctx.conn, ctx.run_id)
    assert (run["status"], run["agent_prompt_version"]) == ("waiting", "flow-v2")  # paused at approve


def test_possible_twice_gets_one_extra_deeper_pass_then_a_flagged_draft(
    ctx, settings, fake
):
    fake["verdicts"] += [POSSIBLE, POSSIBLE]
    final = run_flow("L1", ctx, settings)
    assert final["research_passes"] == 2
    assert fake["judged"] == 2  # the second qualify was NOT the reused 7-day result
    assert 8 in fake["search_k"]  # ...and looked deeper on the extra pass
    extra_questions = fake["asked"][
        1:
    ]  # pass 2 asks exactly about the unknown criteria
    assert extra_questions == [
        "Is Figma a large organization (over 500 employees, or thousands of customer teams or users)?",
        "Does Figma invest in developer tools, platform engineering, or developer experience?",
    ]
    assert final["warnings"] == ["possible fit: scale=unknown, dev_tooling=unknown"]
    assert len(_drafts(ctx)) == 1
    assert (ctx.qualify_k, ctx.reuse_qualification) == (4, True)  # knobs put back


def test_possible_then_strong_drafts_without_a_flag(ctx, settings, fake):
    fake["verdicts"] += [POSSIBLE, STRONG]
    final = run_flow("L1", ctx, settings)
    assert (
        final["research_passes"],
        final["qualification"]["fit"],
        final["warnings"],
    ) == (2, "strong", [])


@pytest.mark.parametrize(
    "verdicts", [[POOR], [POSSIBLE, POOR]], ids=["poor-first", "poor-after-extra-pass"]
)
def test_poor_fit_stops_without_writing_anything(ctx, settings, fake, verdicts):
    fake["verdicts"] += verdicts
    final = run_flow("L1", ctx, settings)
    assert final["outcome"] == "stopped: poor fit"
    assert fake["emails"] == 0 and _drafts(ctx) == []


def test_a_lead_who_replied_stops_before_any_research(ctx, settings, fake):
    d = state.insert_draft(
        ctx.conn, tools.fetch_lead(ctx.conn, "L1"), "Hi", "Hello", None
    )
    state.record_approval(ctx.conn, d.draft_id, True, None)
    send_once(ctx.conn, d, ctx.outbox_dir)
    # now() is the TRANSACTION start time (D4 M4 gotcha): in one test transaction the send and the
    # reply would get the SAME timestamp, and "reply > send" would be false. Backdate the send.
    ctx.conn.execute("UPDATE sends SET sent_at = sent_at - interval '1 day'")
    state.record_reply(ctx.conn, "L1")
    final = run_flow("L1", ctx, settings)
    assert final["outcome"] == "stopped: the lead replied since our last email"
    assert fake["search_k"] == [] and fake["judged"] == 0


def test_a_rejected_draft_ends_in_an_error_not_a_crash(
    ctx, settings, fake, monkeypatch
):
    fake["verdicts"].append(STRONG)
    monkeypatch.setattr(
        flow,
        "write_email",
        lambda lead, sent, findings: EmailDraft(
            subject="Hi", body="Claim at http://x.example/claim"
        ),
    )
    final = run_flow("L1", ctx, settings)
    assert (
        final["outcome"].startswith("error: draft rejected")
        and "links" in final["outcome"]
    )
    assert _drafts(ctx) == []


def test_the_approval_screen_shows_the_possible_fit_flag(ctx, settings, fake, capsys):
    fake["verdicts"] += [POSSIBLE, POSSIBLE]
    run_flow("L1", ctx, settings)
    cli_approve(
        ToolCall("send_email", SendEmailArgs(draft_id="D1")),
        ToolContext(conn=ctx.conn),
        input_fn=lambda p: "n",
    )
    err = capsys.readouterr().err
    assert "⚠ possible fit: scale=unknown, dev_tooling=unknown" in err
    assert err.index("possible fit") < err.index("To: dana.kim@example.com")


# --- the gate still applies to code-driven calls -------------------------------------------


def test_call_tool_validates(ctx):
    with pytest.raises(tools.ToolError, match="invalid arguments"):
        call_tool(ctx, "get_lead", {"lead_id": "L1; DROP TABLE leads"})


def test_call_tool_never_runs_a_tool_that_needs_approval(ctx):
    with pytest.raises(FlowError, match="needs_approval"):
        call_tool(ctx, "send_email", {"draft_id": "D1"})


def test_call_tool_respects_deny(ctx, monkeypatch):
    monkeypatch.setattr(flow, "authorize", lambda call: "deny")
    with pytest.raises(FlowError, match="deny"):
        call_tool(ctx, "get_lead", {"lead_id": "L1"})


# --- D5 M2: Postgres checkpointer, crash → resume -----------------------------------------
# The checkpointer gets its OWN autocommit connection to the test DB, so its rows really commit
# (like after a real crash); the fixture deletes this test's thread afterwards. Business tables
# still sit on the rolled-back `db` connection.


@pytest.fixture
def saver(test_db_url):
    from langgraph.checkpoint.postgres import PostgresSaver

    flow.setup_checkpointer(test_db_url)
    with flow.checkpointer_conn(test_db_url) as conn:
        cp = PostgresSaver(conn)
        threads: list[str] = []
        cp.threads = threads
        yield cp
        for t in threads:
            cp.delete_thread(t)


def _crash_once_in(monkeypatch, tool_name: str):
    """Make the first call to `tool_name` blow up, like the process dying mid-node."""
    real, crashed = flow.call_tool, []

    def call_tool(ctx, name, args):
        if name == tool_name and not crashed:
            crashed.append(name)
            raise RuntimeError("simulated crash")
        return real(ctx, name, args)

    monkeypatch.setattr(flow, "call_tool", call_tool)


def test_crash_in_qualify_then_resume_skips_finished_nodes(ctx, settings, fake, saver, monkeypatch):
    fake["verdicts"].append(STRONG)
    _crash_once_in(monkeypatch, "qualify_lead")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_flow("L1", ctx, settings, checkpointer=saver)
    run_id = ctx.run_id
    saver.threads.append(str(run_id))
    assert state.get_run(ctx.conn, run_id)["status"] == "failed"
    saved = flow.build_graph(ctx, saver).get_state({"configurable": {"thread_id": str(run_id)}})
    assert saved.next == ("qualify",)  # the checkpoint says: research done, qualify next
    assert saved.values["research_passes"] == 1

    final = resume_flow(run_id, ctx, saver)
    assert len(fake["asked"]) == 1  # research did NOT run again
    assert fake["judged"] == 1  # qualify ran (again, from its start)
    assert final["outcome"] == "drafted D1 (awaiting approval)"
    assert [d.draft_id for d in _drafts(ctx)] == ["D1"]
    assert state.get_run(ctx.conn, run_id)["status"] == "waiting"  # ...and now paused at approve


def test_resume_refuses_a_waiting_run_then_a_finished_one(ctx, settings, fake, saver):
    fake["verdicts"].append(STRONG)
    run_flow("L1", ctx, settings, checkpointer=saver)
    saver.threads.append(str(ctx.run_id))
    with pytest.raises(FlowError, match="waiting for approval"):  # --resume can't approve for you
        resume_flow(ctx.run_id, ctx, saver)
    approve_flow("D1", False, ctx, saver)
    with pytest.raises(FlowError, match="already finished"):
        resume_flow(ctx.run_id, ctx, saver)


def test_resume_refuses_a_run_that_is_not_a_flow_run(ctx, saver):
    run_id = state.start_run(ctx.conn, "agent task", "m", "agent-v5")
    with pytest.raises(FlowError, match="not a flow-v2 run"):
        resume_flow(run_id, ctx, saver)


def test_resume_refuses_a_run_with_no_checkpoint(ctx, saver):
    run_id = state.start_run(ctx.conn, "flow L1", "m", flow.FLOW_VERSION)
    state.finish_run(ctx.conn, run_id, "failed")
    with pytest.raises(FlowError, match="no checkpoint"):
        resume_flow(run_id, ctx, saver)


def test_ctrl_c_marks_the_run_failed_and_it_can_be_resumed(ctx, settings, fake, saver, monkeypatch):
    fake["verdicts"].append(STRONG)
    real, hit = flow.call_tool, []

    def call_tool(ctx, name, args):
        if name == "qualify_lead" and not hit:
            hit.append(1)
            raise KeyboardInterrupt  # not an Exception subclass: must still mark the run failed
        return real(ctx, name, args)

    monkeypatch.setattr(flow, "call_tool", call_tool)
    with pytest.raises(KeyboardInterrupt):
        run_flow("L1", ctx, settings, checkpointer=saver)
    saver.threads.append(str(ctx.run_id))
    assert state.get_run(ctx.conn, ctx.run_id)["status"] == "failed"
    assert resume_flow(ctx.run_id, ctx, saver)["outcome"] == "drafted D1 (awaiting approval)"



# --- D5 M3: human approval with interrupt() ------------------------------------------------


def _sends(ctx):
    return ctx.conn.execute("SELECT draft_id, status FROM sends").fetchall()


def _approvals(ctx):
    return ctx.conn.execute("SELECT draft_id, decision FROM approvals ORDER BY approval_id").fetchall()


def _pause_at_approve(ctx, settings, fake, saver):
    fake["verdicts"].append(STRONG)
    final = run_flow("L1", ctx, settings, checkpointer=saver)
    saver.threads.append(str(ctx.run_id))
    return final


def test_the_flow_pauses_at_approve_without_deciding_or_sending(ctx, settings, fake, saver):
    final = _pause_at_approve(ctx, settings, fake, saver)
    snap = flow.build_graph(ctx, saver).get_state({"configurable": {"thread_id": str(ctx.run_id)}})
    assert snap.next == ("approve",)
    assert snap.interrupts[0].value["draft_id"] == "D1"  # what the human is asked about
    assert "approved" not in final
    assert _approvals(ctx) == [] and _sends(ctx) == []
    assert not (ctx.outbox_dir / "D1.eml").exists()


def test_yes_records_the_approval_and_sends_once(ctx, settings, fake, saver):
    _pause_at_approve(ctx, settings, fake, saver)
    final = approve_flow("D1", True, ctx, saver)
    assert final["outcome"] == "sent (written to outbox): D1 to dana.kim@example.com"
    assert _approvals(ctx) == [(1, "approved")]
    assert _sends(ctx) == [(1, "sent")]
    assert (ctx.outbox_dir / "D1.eml").exists()
    assert state.get_run(ctx.conn, ctx.run_id)["status"] == "done"


def test_no_records_the_decline_and_sends_nothing(ctx, settings, fake, saver):
    _pause_at_approve(ctx, settings, fake, saver)
    final = approve_flow("D1", False, ctx, saver)
    assert final["outcome"] == "declined: D1 not sent"
    assert _approvals(ctx) == [(1, "declined")]
    assert _sends(ctx) == [] and not (ctx.outbox_dir / "D1.eml").exists()


def test_approving_twice_is_refused(ctx, settings, fake, saver):
    _pause_at_approve(ctx, settings, fake, saver)
    approve_flow("D1", True, ctx, saver)
    with pytest.raises(FlowError, match="not waiting for approval"):
        approve_flow("D1", True, ctx, saver)
    assert _sends(ctx) == [(1, "sent")]


def test_approve_refuses_an_unknown_draft_or_one_outside_a_flow(ctx, saver):
    with pytest.raises(FlowError, match="no draft D99"):
        approve_flow("D99", True, ctx, saver)
    d = state.insert_draft(ctx.conn, tools.fetch_lead(ctx.conn, "L1"), "Hi", "Hello", None)
    with pytest.raises(FlowError, match=f"no draft {d.draft_id}"):
        approve_flow(d.draft_id, True, ctx, saver)


def test_a_crash_in_send_then_resume_sends_exactly_once(ctx, settings, fake, saver, monkeypatch):
    """The send node RE-RUNS on resume. Crash AFTER the email went out, BEFORE the checkpoint:
    the re-run must find it in the ledger, not send again."""
    _pause_at_approve(ctx, settings, fake, saver)
    real, crashed = flow.execute, []

    def execute(name, args, c):
        out = real(name, args, c)
        if name == "send_email" and not crashed:
            crashed.append(1)
            raise KeyboardInterrupt  # dies right after the send, before LangGraph saves the step
        return out

    monkeypatch.setattr(flow, "execute", execute)
    with pytest.raises(KeyboardInterrupt):
        approve_flow("D1", True, ctx, saver)
    assert state.get_run(ctx.conn, ctx.run_id)["status"] == "failed"
    final = resume_flow(ctx.run_id, ctx, saver)  # send node runs again from its start
    assert final["outcome"] == "already sent: D1 to dana.kim@example.com"
    assert _sends(ctx) == [(1, "sent")]
    assert len(list(ctx.outbox_dir.glob("*.eml"))) == 1
    assert _approvals(ctx) == [(1, "approved")]  # approve did NOT re-run: its checkpoint was saved


def test_the_pause_survives_a_new_process(ctx, settings, fake, saver, test_db_url):
    """'Approve tomorrow': a brand-new saver on a brand-new connection still finds the pause."""
    from langgraph.checkpoint.postgres import PostgresSaver

    _pause_at_approve(ctx, settings, fake, saver)
    with flow.checkpointer_conn(test_db_url) as conn:
        final = approve_flow("D1", False, ctx, PostgresSaver(conn))
    assert final["outcome"] == "declined: D1 not sent"


def test_send_refuses_if_authorize_denies(ctx, settings, fake, saver, monkeypatch):
    _pause_at_approve(ctx, settings, fake, saver)
    monkeypatch.setattr(flow, "authorize", lambda call: "deny")
    with pytest.raises(FlowError, match="decision=deny"):
        approve_flow("D1", True, ctx, saver)
    assert _sends(ctx) == []
