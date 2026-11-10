"""The agent loop with a scripted fake model and an injected authorize (independent of the real one)."""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from research_assistant import tools
from research_assistant.agent import MaxStepsError, cli_approve, run_agent
from research_assistant.config import Settings
from research_assistant.permissions import ToolCall
from research_assistant.schemas import Lead
from research_assistant.state import list_drafts
from research_assistant.tools import ToolContext

DANA = Lead(lead_id="L1", name="Dana Kim", title="VP Engineering", company="Figma", email="dana.kim@example.com")


def _settings(runs_dir: Path) -> Settings:
    return Settings(openai_api_key=SecretStr("sk-test"), model="test-model", runs_dir=runs_dir)


def call(name: str, args: dict | str, call_id: str = "c1") -> dict:
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": raw}


def turn(*calls: dict) -> dict:
    return {"status": "completed", "output": [{"type": "reasoning", "summary": []}, *calls]}


def final(text: str) -> dict:
    return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


class FakeModel:
    """Returns scripted bodies in order; remembers what it was sent."""

    def __init__(self, *bodies: dict):
        self.bodies = list(bodies)
        self.sent: list[list[dict]] = []

    def __call__(self, items: list[dict], tool_defs: list[dict]) -> dict:
        self.sent.append(list(items))
        return self.bodies.pop(0)

    def outputs(self, turn_index: int) -> list[dict]:
        """The function_call_output items in the request for turn `turn_index` (0-based)."""
        return [i for i in self.sent[turn_index] if i.get("type") == "function_call_output"]


def allow_all(call):
    return "allow"


@pytest.fixture
def ctx(state_db, tmp_path):
    """Real test DB (rolled back) with the seed leads; L1 is DANA."""
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


def _drafts(ctx) -> list[str]:
    return [d.draft_id for d in list_drafts(ctx.conn)]


@pytest.fixture
def settings(tmp_path):
    return _settings(tmp_path)


def _runs(settings: Settings) -> list[dict]:
    lines = [line for f in settings.runs_dir.glob("*.jsonl") for line in f.read_text().splitlines()]
    return [json.loads(line) for line in lines]


def test_two_step_run_feeds_tool_result_back(ctx, settings):
    model = FakeModel(turn(call("get_lead", {"lead_id": "L1"})), final("Dana Kim, VP Eng at Figma."))
    result = run_agent("Who is L1?", ctx, respond_fn=model, authorize_fn=allow_all, settings=settings)

    assert result.answer == "Dana Kim, VP Eng at Figma."
    assert result.steps == 2
    [out] = model.outputs(1)
    assert out["call_id"] == "c1"
    assert json.loads(out["output"]) == DANA.model_dump()
    assert {"type": "reasoning", "summary": []} in model.sent[1]  # reasoning went back too


def test_every_call_is_logged(ctx, settings):
    model = FakeModel(turn(call("get_lead", {"lead_id": "L1"})), final("ok"))
    run_agent("t", ctx, respond_fn=model, authorize_fn=allow_all, settings=settings)
    [rec] = [r for r in _runs(settings) if r.get("kind") == "tool_call"]
    assert rec | {"latency_ms": 0} == {
        "ts": rec["ts"],
        "kind": "tool_call",
        "step": 1,
        "call_id": "c1",
        "tool": "get_lead",
        "arguments": '{"lead_id": "L1"}',
        "validated_args": {"lead_id": "L1"},
        "decision": "allow",
        "status": "ok",
        "output_preview": json.dumps(DANA.model_dump()),
        "latency_ms": 0,
    }


def test_parallel_calls_each_get_an_output(ctx, settings):
    model = FakeModel(
        turn(call("get_lead", {"lead_id": "L1"}, "a"), call("get_lead", {"lead_id": "L999"}, "b")),
        final("done"),
    )
    result = run_agent("t", ctx, respond_fn=model, authorize_fn=allow_all, settings=settings)
    outs = {o["call_id"]: json.loads(o["output"]) for o in model.outputs(1)}
    assert outs["a"] == DANA.model_dump()
    assert outs["b"] == {"error": "no lead with id 'L999'"}
    assert [r["status"] for r in result.tool_calls] == ["ok", "error"]


@pytest.mark.parametrize(
    ("decision", "status", "message"),
    [
        ("deny", "denied", "draft_email is not permitted"),
        ("needs_approval", "rejected", "DECLINED: the user said no to draft_email"),  # no approver = no
        ("yes", "denied", "draft_email is not permitted"),  # unexpected value -> fail closed
    ],
)
def test_not_allowed_means_not_executed(ctx, settings, decision, status, message):
    draft = call("draft_email", {"lead_id": "L1", "subject": "Hi", "body": "Hello"})
    model = FakeModel(turn(draft), final("couldn't"))
    result = run_agent("t", ctx, respond_fn=model, authorize_fn=lambda c: decision, settings=settings)

    assert _drafts(ctx) == []  # the tool never ran
    assert message in json.loads(model.outputs(1)[0]["output"])["error"]
    assert result.tool_calls[0]["status"] == status


@pytest.mark.parametrize(
    ("name", "raw"),
    [("delete_lead", '{"lead_id": "L1"}'), ("get_lead", '{"lead_id": "L1", "bcc": "x@evil.test"}')],
)
def test_invalid_calls_never_reach_authorize(ctx, settings, name, raw):
    seen = []
    model = FakeModel(turn(call(name, raw)), final("ok"))
    result = run_agent("t", ctx, respond_fn=model, authorize_fn=lambda c: seen.append(c) or "allow", settings=settings)
    assert seen == []
    assert result.tool_calls[0]["status"] == "invalid"
    assert "error" in json.loads(model.outputs(1)[0]["output"])


def test_authorize_sees_validated_args(ctx, settings):
    seen = []
    model = FakeModel(turn(call("get_lead", {"lead_id": "L1"})), final("ok"))
    run_agent("t", ctx, respond_fn=model, authorize_fn=lambda c: seen.append(c) or "allow", settings=settings)
    [c] = seen
    assert c.name == "get_lead"
    assert c.args == tools.GetLeadArgs(lead_id="L1")


def test_tool_crash_is_reported_without_details(ctx, settings, monkeypatch):
    def boom(conn, query, k):
        raise RuntimeError("password=hunter2 at db host")

    monkeypatch.setattr(tools, "hybrid_search", boom)
    model = FakeModel(turn(call("search_company", {"question": "What does Figma do?"})), final("sorry"))
    result = run_agent("t", ctx, respond_fn=model, authorize_fn=allow_all, settings=settings)

    assert json.loads(model.outputs(1)[0]["output"]) == {"error": "search_company failed (RuntimeError)"}
    assert "hunter2" in result.tool_calls[0]["error"]  # full detail stays in OUR log
    assert result.answer == "sorry"


def test_authorize_crash_stops_the_run(ctx, settings):
    def broken(call):
        raise NotImplementedError

    model = FakeModel(turn(call("draft_email", {"lead_id": "L1", "subject": "s", "body": "b"})), final("x"))
    with pytest.raises(NotImplementedError):
        run_agent("t", ctx, respond_fn=model, authorize_fn=broken, settings=settings)
    assert _drafts(ctx) == []


def test_max_steps(ctx, settings):
    looping = [turn(call("get_lead", {"lead_id": "L1"}, f"c{i}")) for i in range(5)]
    model = FakeModel(*looping)
    with pytest.raises(MaxStepsError, match="after 3 steps"):
        run_agent("t", ctx, max_steps=3, respond_fn=model, authorize_fn=allow_all, settings=settings)
    assert len(model.sent) == 3


# --- approval gate (M3) ---------------------------------------------------------------------


def needs_approval_for_send(call):
    return "needs_approval" if call.name == "send_email" else "allow"


def _draft_then_send():
    return FakeModel(
        turn(call("draft_email", {"lead_id": "L1", "subject": "Hi", "body": "Hello"}, "d")),
        turn(call("send_email", {"draft_id": "D1"}, "s")),
        final("done"),
    )


def test_approved_send_writes_outbox(ctx, settings):
    asked = []
    result = run_agent(
        "draft and send",
        ctx,
        respond_fn=_draft_then_send(),
        authorize_fn=needs_approval_for_send,
        approve_fn=lambda c, x: asked.append(c) or True,
        settings=settings,
    )
    [c] = asked  # asked once, only for send_email, with validated args
    assert (c.name, c.args.draft_id) == ("send_email", "D1")
    assert (ctx.outbox_dir / "D1.eml").exists()
    send = result.tool_calls[-1]
    assert (send["decision"], send["approved"], send["status"]) == ("needs_approval", True, "ok")


@pytest.mark.parametrize("answer", [False, None, "yes", 1])  # only a literal True approves
def test_anything_but_true_is_a_no(ctx, settings, answer):
    model = _draft_then_send()
    result = run_agent(
        "t", ctx, respond_fn=model, authorize_fn=needs_approval_for_send, approve_fn=lambda c, x: answer, settings=settings
    )
    assert not ctx.outbox_dir.exists()
    send = result.tool_calls[-1]
    assert (send["approved"], send["status"]) == (False, "rejected")
    assert "DECLINED: the user said no to send_email" in json.loads(model.outputs(2)[-1]["output"])["error"]


def test_default_is_no_approver(ctx, settings):
    run_agent("t", ctx, respond_fn=_draft_then_send(), authorize_fn=needs_approval_for_send, settings=settings)
    assert not ctx.outbox_dir.exists()


def test_allowed_calls_never_ask(ctx, settings):
    asked = []
    model = FakeModel(turn(call("get_lead", {"lead_id": "L1"})), final("ok"))
    run_agent("t", ctx, respond_fn=model, authorize_fn=allow_all, approve_fn=lambda c, x: asked.append(c), settings=settings)
    assert asked == []


def test_real_policy_end_to_end(ctx, settings):
    """With the REAL authorize: draft runs freely, send waits for the human."""
    asked = []
    run_agent("t", ctx, respond_fn=_draft_then_send(), approve_fn=lambda c, x: asked.append(c.name) or False, settings=settings)
    assert asked == ["send_email"]
    assert _drafts(ctx) == ["D1"]
    assert not ctx.outbox_dir.exists()


# --- cli_approve ----------------------------------------------------------------------------


def _ctx_with_draft(ctx):
    tools.run_tool("draft_email", json.dumps({"lead_id": "L1", "subject": "Hi", "body": "Hello Dana"}), ctx)
    return ctx


SEND_D1 = ToolCall("send_email", tools.SendEmailArgs(draft_id="D1"))


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("y", True), ("Y", True), (" yes ", True), ("", False), ("n", False), ("sure", False), ("yes please", False)],
)
def test_cli_approve_answers(ctx, typed, expected):
    assert cli_approve(SEND_D1, _ctx_with_draft(ctx), input_fn=lambda prompt: typed) is expected


def test_cli_approve_eof_is_no(ctx):
    def eof(prompt):
        raise EOFError

    assert cli_approve(SEND_D1, _ctx_with_draft(ctx), input_fn=eof) is False


def test_cli_approve_shows_the_stored_draft(ctx, capsys):
    cli_approve(SEND_D1, _ctx_with_draft(ctx), input_fn=lambda prompt: "n")
    err = capsys.readouterr().err
    assert "To: dana.kim@example.com" in err
    assert "Subject: Hi" in err
    assert "Hello Dana" in err
