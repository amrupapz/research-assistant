"""D4 M3: the agent's state is in Postgres, and a step + its effect commit together or not at all.

Uses the rolled-back test DB: each `conn.transaction()` in the code is a savepoint here, which
rolls back exactly like a real transaction would.
"""

import pytest
from test_agent import FakeModel, _settings, call, final, turn

from research_assistant import agent, state, tools
from research_assistant.agent import run_agent
from research_assistant.schemas import IcpJudgment
from research_assistant.tools import Tool, ToolContext

DRAFT = {"lead_id": "L1", "subject": "Hi", "body": "Hello Dana"}


@pytest.fixture
def ctx(state_db, tmp_path):
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


@pytest.fixture
def settings(tmp_path):
    return _settings(tmp_path / "runs")


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _run(task, ctx, settings, *bodies, **kw):
    return run_agent(task, ctx, respond_fn=FakeModel(*bodies), settings=settings, **kw)


# --- what gets stored -----------------------------------------------------------------------


def test_run_and_steps_are_recorded_in_order(ctx, settings):
    result = _run("Draft to L1", ctx, settings, turn(call("draft_email", DRAFT, "d")), final("Drafted D1."))

    run = ctx.conn.execute("SELECT task, status, answer, agent_prompt_version FROM runs").fetchone()
    assert run == ("Draft to L1", "done", "Drafted D1.", "agent-v5")
    steps = state.get_steps(ctx.conn, result.run_id)
    assert [(s["seq"], s["turn"], s["kind"], s["tool"], s["status"]) for s in steps] == [
        (1, 1, "model_turn", None, None),
        (2, 1, "tool_call", "draft_email", "ok"),
        (3, 2, "model_turn", None, None),
    ]
    assert steps[0]["payload"][0]["type"] == "reasoning"  # model turns keep reasoning: M4 rebuilds from these
    assert steps[1]["payload"]["validated_args"] == DRAFT
    assert steps[1]["output"]["draft_id"] == "D1"  # exactly what the model was sent back


def test_drafts_belong_to_their_run(ctx, settings):
    first = _run("t", ctx, settings, turn(call("draft_email", DRAFT)), final("ok"))
    second = _run("t", ctx, settings, turn(call("draft_email", DRAFT)), final("ok"))
    assert [d.draft_id for d in state.list_drafts(ctx.conn, first.run_id)] == ["D1"]
    assert [d.draft_id for d in state.list_drafts(ctx.conn, second.run_id)] == ["D2"]
    assert first.run_id != second.run_id


def test_failed_run_is_marked_failed(ctx, settings):
    looping = [turn(call("get_lead", {"lead_id": "L1"}, f"c{i}")) for i in range(3)]
    with pytest.raises(agent.MaxStepsError):
        _run("t", ctx, settings, *looping, max_steps=2)
    assert ctx.conn.execute("SELECT status, finished_at IS NOT NULL FROM runs").fetchone() == ("failed", True)
    assert _count(ctx.conn, "run_steps") == 4  # 2 model turns + 2 tool calls, all kept for debugging/resume


def test_qualification_is_stored_with_its_run(ctx, settings, monkeypatch):
    monkeypatch.setattr(tools, "hybrid_search", lambda conn, q, k: [])
    monkeypatch.setattr(tools, "judge_icp", lambda company, criteria, chunks: IcpJudgment(results=[]))
    result = _run("t", ctx, settings, turn(call("qualify_lead", {"lead_id": "L1"})), final("ok"))
    row = ctx.conn.execute("SELECT lead_id, fit, run_id FROM qualifications").fetchone()
    assert row == ("L1", "possible", result.run_id)  # title yes, evidence unknown -> possible


# --- approvals ------------------------------------------------------------------------------


def _draft_then_send(ctx, settings, approve: bool, draft_id="D1"):
    return _run(
        "t",
        ctx,
        settings,
        turn(call("draft_email", DRAFT, "d")),
        turn(call("send_email", {"draft_id": draft_id}, "s")),
        final("ok"),
        approve_fn=lambda c, x: approve,
    )


@pytest.mark.parametrize(("approve", "decision"), [(True, "approved"), (False, "declined")])
def test_human_answer_is_recorded(ctx, settings, approve, decision):
    result = _draft_then_send(ctx, settings, approve)
    row = ctx.conn.execute("SELECT draft_id, decision, run_id FROM approvals").fetchone()
    assert row == (1, decision, result.run_id)


def test_approval_for_a_missing_draft_records_nothing_and_does_not_crash(ctx, settings):
    result = _draft_then_send(ctx, settings, True, draft_id="D99")
    assert _count(ctx.conn, "approvals") == 0
    assert result.tool_calls[-1]["status"] == "error"  # send_email: no draft with id 'D99'


# --- atomicity: the point of M3 -------------------------------------------------------------


def test_if_the_step_cannot_be_saved_the_draft_is_not_saved_either(ctx, settings, monkeypatch):
    real_add_step = state.add_step

    def add_step(conn, run_id, **kw):
        if kw["kind"] == "tool_call":
            raise RuntimeError("disk full")
        return real_add_step(conn, run_id, **kw)

    monkeypatch.setattr(state, "add_step", add_step)
    with pytest.raises(RuntimeError, match="disk full"):
        _run("t", ctx, settings, turn(call("draft_email", DRAFT)), final("ok"))
    assert _count(ctx.conn, "drafts") == 0  # the effect rolled back with its step: no orphan draft
    assert ctx.conn.execute("SELECT status FROM runs").fetchone() == ("failed",)


def test_tool_that_fails_halfway_leaves_no_partial_writes_but_its_error_is_recorded(ctx, settings, monkeypatch):
    def half_done(args, c):
        tools.draft_email(args, c)  # writes a draft row...
        raise RuntimeError("crashed after the insert")  # ...then dies

    monkeypatch.setitem(tools.REGISTRY, "draft_email", Tool("draft_email", "d", tools.DraftEmailArgs, half_done))
    result = _run("t", ctx, settings, turn(call("draft_email", DRAFT)), final("sorry"))

    assert _count(ctx.conn, "drafts") == 0  # savepoint around the tool undid the half-written draft
    [step] = [s for s in state.get_steps(ctx.conn, result.run_id) if s["kind"] == "tool_call"]
    assert (step["status"], step["output"]) == ("error", {"error": "draft_email failed (RuntimeError)"})


def test_unexpected_decision_is_stored_as_the_deny_that_was_enforced(ctx, settings):
    result = _run("t", ctx, settings, turn(call("draft_email", DRAFT)), final("ok"), authorize_fn=lambda c: "yes")
    [step] = [s for s in state.get_steps(ctx.conn, result.run_id) if s["kind"] == "tool_call"]
    assert (step["decision"], step["status"]) == ("deny", "denied")
    assert result.tool_calls[0]["decision"] == "yes"  # the JSONL log keeps the raw value


def test_human_is_asked_with_no_transaction_block_open(ctx, settings):
    """Never hold a transaction (and its locks) open while a person decides.

    `_num_transactions` is psycopg's private count of open conn.transaction() blocks: test-only use.
    """
    depth_while_asking = []
    _draft_then_send(ctx, settings, True)  # creates D1
    _run(
        "t",
        ToolContext(conn=ctx.conn, outbox_dir=ctx.outbox_dir),
        settings,
        turn(call("send_email", {"draft_id": "D1"})),
        final("ok"),
        approve_fn=lambda c, x: depth_while_asking.append(x.conn._num_transactions) or False,
    )
    assert depth_while_asking == [0]
