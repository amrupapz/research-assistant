"""D4 M4: crash, then `--resume`. Committed steps are replayed from run_steps, never re-run."""

import json

import pytest
from test_agent import FakeModel, _settings, call, final, turn

from research_assistant import state, tools
from research_assistant.agent import (
    MaxStepsError,
    ResumeError,
    SimulatedCrash,
    rebuild,
    resume_agent,
    run_agent,
)
from research_assistant.schemas import ResearchAnswer
from research_assistant.tools import ToolContext

TASK = "Research Figma and draft an intro to L1."
SEARCH = {"company": "Figma", "question": "What does Figma do?"}
DRAFT = {"lead_id": "L1", "subject": "Hi", "body": "Hello Dana"}


@pytest.fixture
def ctx(state_db, tmp_path):
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


@pytest.fixture
def settings(tmp_path):
    return _settings(tmp_path / "runs")


@pytest.fixture
def searches(monkeypatch):
    """Fake search_company backend that counts how often research really runs."""
    calls = []
    answer = ResearchAnswer(
        answer="Figma is a design tool.",
        confidence="medium",
        assumptions=[],
        needs_sources=True,
        citations=[],
    )
    monkeypatch.setattr(
        tools, "hybrid_search", lambda conn, q, k: calls.append(q) or []
    )
    monkeypatch.setattr(tools, "ask", lambda q, chunks: answer)
    return calls


def _crash_after_search(ctx, settings):
    """Turn 1: search. Crash right after its step commits (before the model ever sees the result)."""
    model = FakeModel(turn(call("search_company", SEARCH, "s")))
    with pytest.raises(SimulatedCrash):
        run_agent(
            TASK, ctx, respond_fn=model, settings=settings, crash_after="search_company"
        )
    return ctx.run_id, model


def _fresh(ctx):
    """A new process: new context, same DB."""
    return ToolContext(conn=ctx.conn, outbox_dir=ctx.outbox_dir)


def _status(ctx, run_id):
    return state.get_run(ctx.conn, run_id)["status"]


# --- the crash leaves the right state behind ------------------------------------------------


def test_crash_leaves_run_running_with_research_committed(ctx, settings, searches):
    run_id, _ = _crash_after_search(ctx, settings)
    assert _status(ctx, run_id) == "running"  # a kill doesn't get to say "failed"
    kinds = [(s["kind"], s["tool"]) for s in state.get_steps(ctx.conn, run_id)]
    assert kinds == [("model_turn", None), ("tool_call", "search_company")]
    assert state.list_drafts(ctx.conn) == []


# --- rebuild --------------------------------------------------------------------------------


def test_rebuild_reproduces_what_the_model_was_sent(ctx, settings, searches):
    model = FakeModel(
        turn(call("search_company", SEARCH, "s")),
        turn(call("draft_email", DRAFT, "d")),
        final("ok"),
    )
    result = run_agent(TASK, ctx, respond_fn=model, settings=settings)
    steps = state.get_steps(ctx.conn, result.run_id)
    items, last_turn, pending, answer = rebuild(TASK, steps)

    def normalized(
        xs,
    ):  # jsonb reorders keys inside the output JSON string: compare parsed
        return [
            {**x, "output": json.loads(x["output"])}
            if x.get("type") == "function_call_output"
            else x
            for x in xs
        ]

    assert normalized(items[: len(model.sent[2])]) == normalized(
        model.sent[2]
    )  # exactly the last request
    assert (last_turn, pending, answer) == (3, [], "ok")


def test_rebuild_finds_the_missing_call_of_a_half_done_turn():
    two_calls = turn(
        call("get_lead", {"lead_id": "L1"}, "a"), call("search_company", SEARCH, "b")
    )["output"]
    steps = [
        {
            "kind": "model_turn",
            "turn": 1,
            "payload": two_calls,
            "call_id": None,
            "output": None,
        },
        {
            "kind": "tool_call",
            "turn": 1,
            "payload": {},
            "call_id": "a",
            "output": {"name": "Dana"},
        },
    ]
    items, last_turn, pending, answer = rebuild("t", steps)
    assert [c["call_id"] for c in pending] == ["b"]
    assert (last_turn, answer) == (1, None)
    assert items[-1] == {
        "type": "function_call_output",
        "call_id": "a",
        "output": '{"name": "Dana"}',
    }


def test_rebuild_with_no_steps_starts_from_the_task():
    assert rebuild("t", []) == ([{"role": "user", "content": "t"}], 0, [], None)


# --- resume ---------------------------------------------------------------------------------


def test_resume_continues_to_the_draft_without_redoing_research(
    ctx, settings, searches
):
    run_id, _ = _crash_after_search(ctx, settings)
    assert len(searches) == 1

    model = FakeModel(turn(call("draft_email", DRAFT, "d")), final("Drafted D1."))
    result = resume_agent(run_id, _fresh(ctx), respond_fn=model, settings=settings)

    assert len(searches) == 1  # research was NOT re-run
    assert result.answer == "Drafted D1." and result.run_id == run_id
    assert [d.draft_id for d in state.list_drafts(ctx.conn, run_id)] == ["D1"]
    assert _status(ctx, run_id) == "done"
    first_request = model.sent[0]
    assert first_request[0] == {"role": "user", "content": TASK}
    [out] = [i for i in first_request if i.get("type") == "function_call_output"]
    assert (
        json.loads(out["output"])["answer"] == "Figma is a design tool."
    )  # the STORED result, replayed
    turns = [s["turn"] for s in state.get_steps(ctx.conn, run_id)]
    assert turns == [1, 1, 2, 2, 3]  # numbering continues after the crash


def test_resume_runs_only_the_call_that_never_committed(ctx, settings, searches):
    """Turn 1 proposed get_lead AND search_company; crash after get_lead committed."""
    model = FakeModel(
        turn(
            call("get_lead", {"lead_id": "L1"}, "a"),
            call("search_company", SEARCH, "b"),
        )
    )
    with pytest.raises(SimulatedCrash):
        run_agent(
            TASK, ctx, respond_fn=model, settings=settings, crash_after="get_lead"
        )
    run_id = ctx.run_id
    assert searches == []

    resumed = FakeModel(final("ok"))
    result = resume_agent(run_id, _fresh(ctx), respond_fn=resumed, settings=settings)
    assert [r["tool"] for r in result.tool_calls] == [
        "search_company"
    ]  # get_lead was not re-run
    assert len(searches) == 1
    outs = [
        i["call_id"] for i in resumed.sent[0] if i.get("type") == "function_call_output"
    ]
    assert outs == ["a", "b"]  # every call of turn 1 has its output, in order: no 400


def test_warnings_survive_the_crash(ctx, settings, searches):
    run_id, _ = _crash_after_search(ctx, settings)
    fresh = _fresh(ctx)
    assert fresh.warnings == []
    seen = []
    model = FakeModel(
        turn(call("draft_email", DRAFT, "d")),
        turn(call("send_email", {"draft_id": "D1"}, "x")),
        final("ok"),
    )
    resume_agent(
        run_id,
        fresh,
        respond_fn=model,
        settings=settings,
        approve_fn=lambda c, x: seen.append(list(x.warnings)) or False,
    )
    [warnings] = seen
    assert len(warnings) == 1 and "confidence=medium" in warnings[0]


def test_can_crash_and_resume_twice(ctx, settings, searches):
    run_id, _ = _crash_after_search(ctx, settings)
    with pytest.raises(SimulatedCrash):
        resume_agent(
            run_id,
            _fresh(ctx),
            respond_fn=FakeModel(turn(call("draft_email", DRAFT, "d"))),
            settings=settings,
            crash_after="draft_email",
        )
    result = resume_agent(
        run_id, _fresh(ctx), respond_fn=FakeModel(final("ok")), settings=settings
    )
    assert result.answer == "ok"
    assert [d.draft_id for d in state.list_drafts(ctx.conn)] == [
        "D1"
    ]  # one draft, not two


def test_crash_after_final_answer_saved(ctx, settings, monkeypatch):
    """The final model turn committed, then the process died before finish_run."""
    monkeypatch.setattr(
        state,
        "finish_run",
        lambda *a, **k: (_ for _ in ()).throw(SimulatedCrash("died")),
    )
    with pytest.raises(SimulatedCrash):
        run_agent(
            TASK, ctx, respond_fn=FakeModel(final("All done.")), settings=settings
        )
    monkeypatch.undo()
    no_model = FakeModel()  # resume must not need the model at all
    result = resume_agent(
        ctx.run_id, _fresh(ctx), respond_fn=no_model, settings=settings
    )
    assert result.answer == "All done." and no_model.sent == []
    assert _status(ctx, ctx.run_id) == "done"


def test_failed_run_can_resume_with_more_steps(ctx, settings):
    looping = [turn(call("get_lead", {"lead_id": "L1"}, f"c{i}")) for i in range(2)]
    with pytest.raises(MaxStepsError):
        run_agent(
            TASK, ctx, respond_fn=FakeModel(*looping), settings=settings, max_steps=2
        )
    run_id = ctx.run_id
    assert _status(ctx, run_id) == "failed"
    result = resume_agent(
        run_id,
        _fresh(ctx),
        respond_fn=FakeModel(final("ok")),
        settings=settings,
        max_steps=3,
    )
    assert (result.answer, result.steps) == ("ok", 3)  # max_steps counts old turns too


# --- refusals -------------------------------------------------------------------------------


def test_done_run_is_refused(ctx, settings):
    result = run_agent(TASK, ctx, respond_fn=FakeModel(final("ok")), settings=settings)
    with pytest.raises(ResumeError, match="already finished"):
        resume_agent(
            result.run_id, _fresh(ctx), respond_fn=FakeModel(), settings=settings
        )


def test_unknown_run_is_refused(ctx, settings):
    import uuid

    with pytest.raises(ResumeError, match="no run"):
        resume_agent(
            uuid.uuid4(), _fresh(ctx), respond_fn=FakeModel(), settings=settings
        )


def test_run_from_an_older_prompt_is_refused(ctx, settings, searches):
    run_id, _ = _crash_after_search(ctx, settings)
    ctx.conn.execute(
        "UPDATE runs SET agent_prompt_version = 'agent-v3' WHERE run_id = %s", [run_id]
    )
    with pytest.raises(ResumeError, match="agent-v3"):
        resume_agent(run_id, _fresh(ctx), respond_fn=FakeModel(), settings=settings)
    assert _status(ctx, run_id) == "running"  # refused before touching anything


def test_run_from_another_model_is_refused(ctx, settings, searches):
    run_id, _ = _crash_after_search(ctx, settings)
    other = settings.model_copy(update={"model": "other-model"})
    with pytest.raises(ResumeError, match="other-model"):
        resume_agent(run_id, _fresh(ctx), respond_fn=FakeModel(), settings=other)
