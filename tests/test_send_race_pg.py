"""D4 M7: two live processes sending the same draft at once. Forced interleaving, no sleeps.

These need COMMITTED data that two connections can both see, so they can't use the rolled-back
`db` fixture. They commit to the test DB and clean up after themselves.
"""

import threading

import psycopg
import pytest
from conftest import AGENT_TABLES

from research_assistant import sending, state
from research_assistant.db import connect, seed_leads
from research_assistant.schemas import Lead
from research_assistant.sending import SendError, send_once

LEAD = Lead(lead_id="L1", name="Dana Kim", title="VP Engineering", company="Figma", email="dana.kim@example.com")


@pytest.fixture
def committed(test_db_url):
    """One approved draft, committed. Two autocommit connections: process A and process B."""

    def clean():
        with connect(test_db_url, autocommit=True) as c:
            c.execute(f"TRUNCATE {AGENT_TABLES} RESTART IDENTITY")
            c.execute("DELETE FROM leads")

    clean()
    with connect(test_db_url, autocommit=True) as c:
        seed_leads(c)
        draft = state.insert_draft(c, LEAD, "Hi", "Hello Dana", None)
        state.record_approval(c, draft.draft_id, True, None)
    a, b = connect(test_db_url, autocommit=True), connect(test_db_url, autocommit=True)
    yield draft, a, b
    for conn in (a, b):
        if not conn.closed:
            conn.close()
    clean()


def _pause_a_inside_deliver(monkeypatch, a_thread_name, *, after_writing: bool):
    """Process A stops inside deliver (before or after writing the file) until released."""
    inside, release = threading.Event(), threading.Event()
    real = sending.deliver

    def deliver(outbox_dir, draft, key):
        if threading.current_thread().name != a_thread_name:
            return real(outbox_dir, draft, key)
        where = real(outbox_dir, draft, key) if after_writing else None
        inside.set()
        assert release.wait(5), "test never released A"
        return where if after_writing else real(outbox_dir, draft, key)

    monkeypatch.setattr(sending, "deliver", deliver)
    return inside, release


def _start_a(a_conn, draft, outbox, result: dict) -> threading.Thread:
    def run():
        try:
            result["value"] = send_once(a_conn, draft, outbox)
        except Exception as e:  # noqa: BLE001 -- recorded for the assertions
            result["error"] = e

    t = threading.Thread(target=run, name="process-A")
    t.start()
    return t


def test_two_live_senders_only_one_sends(committed, tmp_path, monkeypatch):
    draft, a, b = committed
    outbox = tmp_path / "outbox"
    inside, release = _pause_a_inside_deliver(monkeypatch, "process-A", after_writing=False)
    result: dict = {}
    t = _start_a(a, draft, outbox, result)
    assert inside.wait(5)  # A has claimed, holds the row lock, and is about to send

    with pytest.raises(SendError, match="another process is sending"):
        send_once(b, draft, outbox)  # B sees pending + not delivered -- but can't get the lock
    assert list(outbox.glob("*.eml")) == []

    release.set()
    t.join(5)
    assert result["value"]["status"] == "sent (written to outbox)"
    assert [p.name for p in outbox.glob("*.eml")] == ["D1.eml"]
    assert send_once(b, draft, outbox)["status"] == "already sent"  # B retries later: nothing to do


def test_dead_sender_releases_the_lock_and_retry_recovers_without_resending(committed, tmp_path, monkeypatch):
    """A writes the email, then its process is killed before recording it. Postgres drops A's lock
    with its connection; B gets the lock, sees the email already went out, and only records it."""
    draft, a, b = committed
    outbox = tmp_path / "outbox"
    inside, release = _pause_a_inside_deliver(monkeypatch, "process-A", after_writing=True)
    result: dict = {}
    t = _start_a(a, draft, outbox, result)
    assert inside.wait(5)
    sent_bytes = (outbox / "D1.eml").read_bytes()

    killed = b.execute("SELECT pg_terminate_backend(%s, 5000)", [a.info.backend_pid]).fetchone()[0]
    assert killed  # waited until A's backend is gone (and its lock with it)
    release.set()
    t.join(5)
    assert isinstance(result.get("error"), psycopg.OperationalError)  # A died before mark_sent
    assert state.get_send(b, draft.draft_id)["status"] == "pending"

    assert send_once(b, draft, outbox)["status"] == "already sent (recovered)"
    assert (outbox / "D1.eml").read_bytes() == sent_bytes  # not re-sent
    assert state.get_send(b, draft.draft_id)["status"] == "sent"
