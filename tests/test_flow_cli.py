"""D7 M6: `flow --resume RUN_ID --tenant acme`. Tenant isolation hides other tenants' runs from a
connection, so an operator must be able to say whose run it is (RUNBOOK: resume a stuck run)."""

import sys
import uuid
from contextlib import nullcontext

import pytest

from research_assistant import flow

RUN = uuid.uuid4()


@pytest.fixture
def seen(monkeypatch):
    got = {}

    def fake_connect(autocommit=False, tenant="default", url=None):
        got["tenant"] = tenant
        return nullcontext(object())

    class Settings:
        database_url = "postgresql://unused"

    monkeypatch.setattr("research_assistant.db.connect", fake_connect)
    monkeypatch.setattr(flow, "checkpointer_conn", lambda url: nullcontext(object()))
    monkeypatch.setattr(flow, "get_settings", lambda: Settings)
    monkeypatch.setattr("langgraph.checkpoint.postgres.PostgresSaver", lambda conn: object())
    monkeypatch.setattr(flow, "resume_flow", lambda run_id, ctx, saver: got.update(run=run_id) or {"outcome": "ok"})
    return got


def test_resume_uses_the_given_tenant(seen, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["flow", "--resume", str(RUN), "--tenant", "acme"])
    flow._main()
    assert seen == {"tenant": "acme", "run": RUN}


def test_the_default_tenant_is_the_cli_tenant(seen, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["flow", "--resume", str(RUN)])
    flow._main()
    assert seen["tenant"] == "default"
