# ruff: noqa: F811  (asks/judges/ctx are pytest fixtures imported from test_memory_pg and used by name)
"""D7 M2: tenant isolation in the DATA layer (no HTTP). The connection carries the tenant; every
query filters on it. 'default' owns the seed leads; 'acme' gets its own lead at the same company."""

import psycopg
import pytest
from test_memory_pg import _search, asks, ctx, judges  # noqa: F401

from research_assistant import auth, state
from research_assistant.db import connect, fetch_lead, seed_leads, set_tenant
from research_assistant.schemas import Lead
from research_assistant.tools import run_tool

ACME_LEAD = Lead(lead_id="L9", name="Sam Roe", title="CTO", company="Figma", email="sam.roe@example.com")


@pytest.fixture
def acme(ctx):
    """Tenant 'acme' with one lead at Figma (same company as default's L1). Leaves the conn on 'default'."""
    ctx.conn.execute("INSERT INTO tenants (tenant_id) VALUES ('acme') ON CONFLICT DO NOTHING")
    set_tenant(ctx.conn, "acme")
    seed_leads(ctx.conn, [ACME_LEAD])
    set_tenant(ctx.conn, "default")
    return ctx


def test_tenant_b_does_not_reuse_tenant_a_research(acme, asks):
    _search(acme)  # default researches Figma: 1 model call
    set_tenant(acme.conn, "acme")
    assert state.reusable_finding(acme.conn, "Figma", "What does Figma do?", "any") is None
    assert state.recent_findings(acme.conn, "Figma") == []  # can't even see THAT default looked at Figma
    _search(acme)
    assert len(asks) == 2  # acme paid for its own answer; nothing was shared
    set_tenant(acme.conn, "default")
    _search(acme)
    assert len(asks) == 2  # default still reuses ITS answer (reuse works within a tenant)


def test_tenant_b_does_not_see_tenant_a_qualification_or_outreach(acme, judges):
    run_tool("qualify_lead", '{"lead_id": "L1"}', acme)
    set_tenant(acme.conn, "acme")
    assert state.latest_qualification(acme.conn, "L1") is None
    status = run_tool("lead_status", '{"lead_id": "L9"}', acme)
    assert status["latest_qualification"] is None and status["recent_research"] == []


def test_another_tenants_lead_does_not_exist(acme):
    assert fetch_lead(acme.conn, "L9") is None  # we're 'default'
    assert state.owner(acme.conn, "lead", "L9") is None
    assert state.owner(acme.conn, "lead", "L1") == "default"
    set_tenant(acme.conn, "acme")
    assert fetch_lead(acme.conn, "L9") == ACME_LEAD and fetch_lead(acme.conn, "L1") is None


def test_seeding_cannot_overwrite_another_tenants_lead(acme):
    set_tenant(acme.conn, "acme")
    seed_leads(acme.conn, [Lead(lead_id="L1", name="Evil", title="x", company="x", email="evil@example.com")])
    set_tenant(acme.conn, "default")
    assert fetch_lead(acme.conn, "L1").name == "Dana Kim"


def test_inserts_land_in_the_connections_tenant(acme):
    set_tenant(acme.conn, "acme")
    run_id = state.start_run(acme.conn, "t", "m", "v")
    assert acme.conn.execute("SELECT tenant_id FROM runs WHERE run_id = %s", [run_id]).fetchone()[0] == "acme"
    set_tenant(acme.conn, "default")
    assert state.get_run(acme.conn, run_id) is None
    assert state.claim_waiting(acme.conn, run_id) is False  # updates are filtered too


def test_no_tenant_set_fails_closed(test_db_url):
    with connect(test_db_url, tenant=None) as conn, pytest.raises(psycopg.errors.UndefinedObject):
        fetch_lead(conn, "L1")  # an error, not every tenant's rows


# --- keys -----------------------------------------------------------------------------------


def test_only_the_hash_of_a_key_is_stored(db):
    key_id, key = auth.create_key(db, "default", "member", "laptop")
    assert key.startswith("ra_") and len(key) > 40
    stored = db.execute("SELECT key_sha256 FROM api_keys WHERE key_id = %s", [key_id]).fetchone()[0]
    assert stored == auth.hash_key(key) and key not in stored
    assert auth.find_principal(db, key) == auth.Principal(key_id, "default", "member")


def test_a_revoked_or_unknown_key_finds_nobody(db):
    key_id, key = auth.create_key(db, "default", "approver")
    assert auth.revoke_key(db, key_id) is True
    assert auth.find_principal(db, key) is None
    assert auth.find_principal(db, "ra_not-a-real-key") is None
    assert auth.revoke_key(db, key_id) is False  # already revoked


@pytest.fixture
def cli_db(test_db_url, monkeypatch):
    """The keys CLI's `with connect() as conn:` gets ONE connection that never commits; rolled back after."""
    conn = connect(test_db_url, tenant=None)

    class NoCommit:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("research_assistant.db.connect", lambda **kw: NoCommit())
    yield conn
    conn.rollback()
    conn.close()


def test_keys_cli_creates_and_lists_without_showing_the_key_or_hash(cli_db, capsys):
    auth.main(["create-key", "--tenant", "default", "--role", "member", "--label", "cli-test"])
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert key.startswith("ra_")
    auth.main(["list-keys"])
    out = capsys.readouterr().out
    assert "cli-test" in out and key not in out and auth.hash_key(key) not in out


def test_keys_cli_rejects_an_unknown_tenant(cli_db):
    with pytest.raises(SystemExit, match="no tenant nope"):
        auth.main(["create-key", "--tenant", "nope", "--role", "member"])
