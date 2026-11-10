"""D4 M5: send_once. Crash at every point between CLAIM, SEND and RECORD; a retry must never
send twice and must never lose a send. Rolled-back test DB; crashes are simulated."""

import pytest
from test_agent import FakeModel, _settings, call, final, turn

from research_assistant import sending, state
from research_assistant.agent import SimulatedCrash, resume_agent, run_agent
from research_assistant.schemas import Lead
from research_assistant.sending import (
    SendError,
    idempotency_key,
    send_once,
    was_delivered,
)
from research_assistant.tools import ToolContext

LEAD = Lead(
    lead_id="L1",
    name="Dana Kim",
    title="VP Engineering",
    company="Figma",
    email="dana.kim@example.com",
)


@pytest.fixture
def conn(state_db):
    return state_db


@pytest.fixture
def outbox(tmp_path):
    return tmp_path / "outbox"


@pytest.fixture
def draft(conn):
    d = state.insert_draft(conn, LEAD, "Hi", "Hello Dana", None)
    state.record_approval(conn, d.draft_id, True, None)
    return d


def _emails(outbox) -> list[str]:
    return sorted(p.name for p in outbox.glob("*.eml")) if outbox.exists() else []


def _crash(*a, **k):
    raise SimulatedCrash("power cut")


# --- the happy path -------------------------------------------------------------------------


def test_sends_once_and_records_it(conn, draft, outbox):
    out = send_once(conn, draft, outbox)
    assert out["status"] == "sent (written to outbox)"
    row = state.get_send(conn, draft.draft_id)
    assert (row["status"], row["idempotency_key"], row["channel"]) == (
        "sent",
        "draft-1",
        "outbox",
    )
    assert row["sent_at"] is not None
    assert was_delivered(outbox, draft, "draft-1")  # the email carries its key


def test_second_call_does_not_send_again(conn, draft, outbox):
    send_once(conn, draft, outbox)
    assert send_once(conn, draft, outbox)["status"] == "already sent"
    assert _emails(outbox) == ["D1.eml"]


def test_key_comes_from_the_draft_not_from_chance():
    assert idempotency_key("D7") == idempotency_key("D7") == "draft-7"
    assert idempotency_key("D7") != idempotency_key("D8")


# --- approval is checked in CODE (the FK can't) ---------------------------------------------


def test_no_approval_no_send(conn, outbox):
    d = state.insert_draft(conn, LEAD, "Hi", "x", None)
    with pytest.raises(SendError, match="no approval"):
        send_once(conn, d, outbox)
    assert _emails(outbox) == [] and state.get_send(conn, d.draft_id) is None


def test_declined_no_send(conn, outbox):
    d = state.insert_draft(conn, LEAD, "Hi", "x", None)
    state.record_approval(conn, d.draft_id, False, None)
    with pytest.raises(SendError, match="no approval"):
        send_once(conn, d, outbox)


def test_latest_answer_wins(conn, outbox):
    """Said yes, then no (e.g. on a later run): the NO counts."""
    d = state.insert_draft(conn, LEAD, "Hi", "x", None)
    state.record_approval(conn, d.draft_id, True, None)
    state.record_approval(conn, d.draft_id, False, None)
    with pytest.raises(SendError):
        send_once(conn, d, outbox)


# --- crash at every point -------------------------------------------------------------------


def test_crash_between_claim_and_send_then_retry_sends_once(
    conn, draft, outbox, monkeypatch
):
    monkeypatch.setattr(sending, "deliver", _crash)
    with pytest.raises(SimulatedCrash):
        send_once(conn, draft, outbox)
    assert (
        state.get_send(conn, draft.draft_id)["status"] == "pending"
    )  # the claim was saved
    assert _emails(outbox) == []  # but nothing went out
    monkeypatch.undo()

    assert send_once(conn, draft, outbox)["status"] == "sent (written to outbox)"
    assert _emails(outbox) == ["D1.eml"]
    assert state.get_send(conn, draft.draft_id)["status"] == "sent"


def test_crash_between_send_and_record_then_retry_does_not_send_again(
    conn, draft, outbox, monkeypatch
):
    """THE case M5 exists for: the email left, but the ledger still says pending."""
    monkeypatch.setattr(state, "mark_sent", _crash)
    with pytest.raises(SimulatedCrash):
        send_once(conn, draft, outbox)
    assert _emails(outbox) == ["D1.eml"]  # it went out...
    assert (
        state.get_send(conn, draft.draft_id)["status"] == "pending"
    )  # ...but we never recorded it
    monkeypatch.undo()

    sent_files = [p.read_bytes() for p in outbox.glob("*.eml")]
    out = send_once(conn, draft, outbox)
    assert out["status"] == "already sent (recovered)"
    assert [
        p.read_bytes() for p in outbox.glob("*.eml")
    ] == sent_files  # untouched, not re-sent
    assert state.get_send(conn, draft.draft_id)["status"] == "sent"


def test_pending_on_a_channel_that_cant_check_is_refused(
    conn, draft, outbox, monkeypatch
):
    """Real email can't answer 'did key X go out?'. Then: refuse, a human checks. Never guess."""
    monkeypatch.setattr(sending, "deliver", _crash)
    with pytest.raises(SimulatedCrash):
        send_once(conn, draft, outbox, channel="smtp")
    monkeypatch.undo()
    with pytest.raises(SendError, match="check manually"):
        send_once(conn, draft, outbox, channel="smtp")
    assert _emails(outbox) == []


def test_half_written_file_never_looks_sent(conn, draft, outbox, monkeypatch):
    """Crash while writing: the temp file is left behind, the real .eml never appears."""
    monkeypatch.setattr(sending.os, "replace", _crash)
    with pytest.raises(SimulatedCrash):
        send_once(conn, draft, outbox)
    monkeypatch.undo()
    assert _emails(outbox) == []
    assert send_once(conn, draft, outbox)["status"] == "sent (written to outbox)"
    assert _emails(outbox) == ["D1.eml"]


def test_wont_overwrite_someone_elses_file(conn, draft, outbox):
    outbox.mkdir(parents=True)
    (outbox / "D1.eml").write_text("an old file from P3")
    with pytest.raises(SendError, match="refusing to overwrite"):
        send_once(conn, draft, outbox)
    assert (outbox / "D1.eml").read_text() == "an old file from P3"


# --- through the real agent: crash, resume, exactly one email -------------------------------


def test_agent_crash_mid_send_then_resume_sends_exactly_once(
    conn, outbox, tmp_path, monkeypatch
):
    settings = _settings(tmp_path / "runs")
    ctx = ToolContext(conn=conn, outbox_dir=outbox)
    monkeypatch.setattr(
        state, "mark_sent", _crash
    )  # dies after the email left, before it's recorded
    model = FakeModel(
        turn(
            call(
                "draft_email",
                {"lead_id": "L1", "subject": "Hi", "body": "Hello Dana"},
                "d",
            )
        ),
        turn(call("send_email", {"draft_id": "D1"}, "s")),
    )
    with pytest.raises(SimulatedCrash):
        run_agent(
            "draft and send",
            ctx,
            respond_fn=model,
            approve_fn=lambda c, x: True,
            settings=settings,
        )
    monkeypatch.undo()
    assert _emails(outbox) == ["D1.eml"]

    # The send_email step never committed, so resume sees it as pending and runs it again,
    # and the human is asked again. send_once must notice the email already left.
    resumed = resume_agent(
        ctx.run_id,
        ToolContext(conn=conn, outbox_dir=outbox),
        respond_fn=FakeModel(final("Sent.")),
        approve_fn=lambda c, x: True,
        settings=settings,
    )
    [send] = resumed.tool_calls
    assert (send["tool"], send["status"]) == ("send_email", "ok")
    assert _emails(outbox) == ["D1.eml"]  # exactly one email
    assert state.get_send(conn, "D1")["status"] == "sent"


def test_declined_through_the_agent_records_no_send(conn, outbox, tmp_path):
    ctx = ToolContext(conn=conn, outbox_dir=outbox)
    model = FakeModel(
        turn(
            call(
                "draft_email", {"lead_id": "L1", "subject": "Hi", "body": "Hello"}, "d"
            )
        ),
        turn(call("send_email", {"draft_id": "D1"}, "s")),
        final("ok"),
    )
    run_agent(
        "t",
        ctx,
        respond_fn=model,
        approve_fn=lambda c, x: False,
        settings=_settings(tmp_path / "runs"),
    )
    assert state.get_send(conn, "D1") is None
    assert state.latest_approval(conn, "D1")[1] == "declined"


# --- the approval screen tells the truth about the ledger (M5 fix after the live run) -------


def _screen(conn, outbox, capsys) -> str:
    from research_assistant.agent import cli_approve
    from research_assistant.permissions import ToolCall
    from research_assistant.tools import SendEmailArgs

    ctx = ToolContext(conn=conn, outbox_dir=outbox)
    cli_approve(
        ToolCall("send_email", SendEmailArgs(draft_id="D1")),
        ctx,
        input_fn=lambda p: "n",
    )
    return capsys.readouterr().err


def test_screen_says_already_sent(conn, draft, outbox, capsys):
    send_once(conn, draft, outbox)
    err = _screen(conn, outbox, capsys)
    assert "ALREADY SENT" in err and "will NOT send it again" in err
    assert err.index("ALREADY SENT") < err.index(
        "To: dana.kim@example.com"
    )  # before the draft


def test_screen_flags_an_unfinished_send(conn, draft, outbox, capsys, monkeypatch):
    monkeypatch.setattr(sending, "deliver", _crash)
    with pytest.raises(SimulatedCrash):
        send_once(conn, draft, outbox)
    monkeypatch.undo()
    assert "did not finish" in _screen(conn, outbox, capsys)


def test_screen_is_quiet_for_a_fresh_draft(conn, draft, outbox, capsys):
    assert "⚠" not in _screen(conn, outbox, capsys)


# --- a decline must not erase an earlier send (M5 fix after the second live run) ------------


def _decline_send(conn, outbox, tmp_path):
    import json

    model = FakeModel(turn(call("send_email", {"draft_id": "D1"}, "s")), final("ok"))
    run_agent(
        "Send D1.",
        ToolContext(conn=conn, outbox_dir=outbox),
        respond_fn=model,
        approve_fn=lambda c, x: False,
        settings=_settings(tmp_path / "runs"),
    )
    return json.loads(model.outputs(1)[0]["output"])["error"]


def test_declining_an_already_sent_draft_tells_the_model_it_was_sent(
    conn, draft, outbox, tmp_path
):
    send_once(conn, draft, outbox)
    error = _decline_send(conn, outbox, tmp_path)
    assert error.startswith("DECLINED")
    assert "ALREADY SENT earlier" in error and "unaffected" in error
    assert (
        state.get_send(conn, "D1")["status"] == "sent"
    )  # the decline changed nothing in the ledger


def test_declining_an_unsent_draft_has_no_note(conn, draft, outbox, tmp_path):
    error = _decline_send(conn, outbox, tmp_path)
    assert error.startswith("DECLINED") and "ALREADY SENT" not in error
