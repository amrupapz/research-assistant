"""D4 M2: migrations go up and down cleanly, and the constraints from the M0 sketch really fire.

Up/down tests COMMIT schema changes to the test DB; they always finish back at head.
Constraint tests use the rolled-back `db` fixture, one savepoint per expected failure.
"""

import psycopg
import pytest
from alembic import command
from conftest import TEST_DIMS

from research_assistant.db import alembic_config, connect, migrate

BASELINE = {"chunks", "leads"}
AGENT_STATE = {"runs", "run_steps", "research_findings", "finding_citations", "qualifications", "drafts", "approvals", "sends"}
MEMORY = {"replies"}


def _tables(url: str) -> set[str]:
    with connect(url) as conn:
        rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
    # LangGraph's checkpoint tables (D5 M2) live in the same DB but aren't Alembic's: ignore them.
    return {r[0] for r in rows if not r[0].startswith("checkpoint")} - {"alembic_version"}


def _columns(url: str, table: str) -> set[str]:
    with connect(url) as conn:
        rows = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", [table]).fetchall()
    return {r[0] for r in rows}


def _revision(url: str) -> str | None:
    with connect(url) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    return row[0] if row else None


@pytest.fixture
def at_head_afterwards(test_db_url):
    yield test_db_url
    migrate(test_db_url, dims=TEST_DIMS)  # even if the test failed halfway


def test_up_and_down_step_by_step(at_head_afterwards):
    url, cfg = at_head_afterwards, alembic_config(at_head_afterwards, TEST_DIMS)
    assert _tables(url) == BASELINE | AGENT_STATE | MEMORY
    assert _revision(url) == "0005"
    assert "trace_id" in _columns(url, "runs")

    command.downgrade(cfg, "-1")  # 0005 only adds runs.trace_id: same tables
    assert _revision(url) == "0004"
    assert "trace_id" not in _columns(url, "runs")

    command.downgrade(cfg, "-1")  # 0004 only changes a CHECK: same tables
    assert _tables(url) == BASELINE | AGENT_STATE | MEMORY
    assert _revision(url) == "0003"

    command.downgrade(cfg, "-1")
    assert _tables(url) == BASELINE | AGENT_STATE
    assert _revision(url) == "0002"

    command.downgrade(cfg, "-1")
    assert _tables(url) == BASELINE
    assert _revision(url) == "0001"

    command.downgrade(cfg, "base")
    assert _tables(url) == set()
    assert _revision(url) is None

    command.upgrade(cfg, "head")
    assert _tables(url) == BASELINE | AGENT_STATE | MEMORY


def test_upgrade_is_a_no_op_at_head(at_head_afterwards):
    migrate(at_head_afterwards, dims=TEST_DIMS)
    migrate(at_head_afterwards, dims=TEST_DIMS)
    assert _revision(at_head_afterwards) == "0005"


def test_baseline_uses_the_configured_dims(at_head_afterwards):
    with connect(at_head_afterwards) as conn:
        typ = conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        ).fetchone()[0]
    assert typ == f"vector({TEST_DIMS})"


# --- constraints ----------------------------------------------------------------------------


def _lead(db, lead_id="L1"):
    db.execute("INSERT INTO leads VALUES (%s, 'Dana Kim', 'VP Engineering', 'Figma', 'dana.kim@example.com')", [lead_id])


def _draft(db, lead_id="L1") -> int:
    return db.execute(
        "INSERT INTO drafts (lead_id, to_email, subject, body) VALUES (%s, 'dana.kim@example.com', 'Hi', 'Hello') "
        "RETURNING draft_id",
        [lead_id],
    ).fetchone()[0]


def _approval(db, draft_id, decision="approved") -> int:
    return db.execute(
        "INSERT INTO approvals (draft_id, decision) VALUES (%s, %s) RETURNING approval_id", [draft_id, decision]
    ).fetchone()[0]


def _send(db, draft_id, approval_id, key):
    db.execute(
        "INSERT INTO sends (draft_id, approval_id, idempotency_key, channel) VALUES (%s, %s, %s, 'outbox')",
        [draft_id, approval_id, key],
    )


def _fails(db, exc, sql, params=()):
    with pytest.raises(exc), db.transaction():  # savepoint: the outer test transaction survives
        db.execute(sql, params)


def test_draft_needs_an_existing_lead(db):
    _fails(db, psycopg.errors.ForeignKeyViolation, "INSERT INTO drafts (lead_id, to_email, subject, body) VALUES ('L9', 'x', 's', 'b')")


def test_deleting_a_lead_with_drafts_is_refused(db):
    _lead(db)
    _draft(db)
    _fails(db, psycopg.errors.ForeignKeyViolation, "DELETE FROM leads WHERE lead_id = 'L1'")  # RESTRICT


def test_second_send_for_the_same_draft_is_refused(db):
    _lead(db)
    d = _draft(db)
    a = _approval(db, d)
    _send(db, d, a, "key-1")
    with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
        _send(db, d, a, "key-2")  # different key, same draft: still refused


def test_idempotency_key_is_unique(db):
    _lead(db)
    d1, d2 = _draft(db), _draft(db)
    _send(db, d1, _approval(db, d1), "same-key")
    with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
        _send(db, d2, _approval(db, d2), "same-key")


def test_send_needs_an_approval(db):
    _lead(db)
    d = _draft(db)
    _fails(
        db,
        psycopg.errors.NotNullViolation,
        "INSERT INTO sends (draft_id, idempotency_key, channel) VALUES (%s, 'k', 'outbox')",
        [d],
    )


def test_a_declined_approval_still_satisfies_the_fk(db):
    """Documents the LIMIT from M0: the FK only checks the approval EXISTS, not that it says yes.
    Checking 'approved' is the code's job (send_once, M5)."""
    _lead(db)
    d = _draft(db)
    _send(db, d, _approval(db, d, "declined"), "k")  # the DB allows it


def test_sent_status_needs_sent_at(db):
    _lead(db)
    d = _draft(db)
    _fails(
        db,
        psycopg.errors.CheckViolation,
        "INSERT INTO sends (draft_id, approval_id, idempotency_key, channel, status) VALUES (%s, %s, 'k', 'outbox', 'sent')",
        [d, _approval(db, d)],
    )


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        ("INSERT INTO qualifications (lead_id, fit, criteria, icp_version) VALUES ('L1', 'great', '[]', 'v')", ()),
        ("INSERT INTO approvals (draft_id, decision) VALUES (%s, 'maybe')", "draft"),
        ("INSERT INTO sends (draft_id, approval_id, idempotency_key, channel) VALUES (%s, %s, 'k', 'linkedin')", "send"),
        ("INSERT INTO runs (task, model, agent_prompt_version, status) VALUES ('t', 'm', 'v', 'paused')", ()),
    ],
    ids=["fit", "decision", "channel", "run_status"],
)
def test_check_constraints(db, sql, params):
    _lead(db)
    d = _draft(db)
    params = {"draft": [d], "send": [d, _approval(db, d)]}.get(params, params)
    _fails(db, psycopg.errors.CheckViolation, sql, params)


def test_deleting_a_finding_deletes_its_citations(db):
    fid = db.execute(
        "INSERT INTO research_findings (company, question, answer, confidence, needs_sources, prompt_version) "
        "VALUES ('Figma', 'q', 'a', 'high', false, 'v4') RETURNING finding_id"
    ).fetchone()[0]
    db.execute(
        "INSERT INTO finding_citations VALUES (%s, 'figma-about#000', 'https://figma.com/about', '2026-11-01')", [fid]
    )
    db.execute("DELETE FROM research_findings WHERE finding_id = %s", [fid])  # CASCADE: part-of
    assert db.execute("SELECT count(*) FROM finding_citations").fetchone()[0] == 0


def test_citations_do_not_reference_chunks(db):
    """Snapshot, not FK: a citation to a chunk that no longer exists is fine."""
    fid = db.execute(
        "INSERT INTO research_findings (company, question, answer, confidence, needs_sources, prompt_version) "
        "VALUES ('Figma', 'q', 'a', 'high', false, 'v4') RETURNING finding_id"
    ).fetchone()[0]
    db.execute("INSERT INTO finding_citations VALUES (%s, 'deleted#000', 'https://x.example', '2020-01-01')", [fid])


def test_tool_call_step_needs_call_id_and_tool(db):
    run_id = db.execute(
        "INSERT INTO runs (task, model, agent_prompt_version) VALUES ('t', 'm', 'agent-v4') RETURNING run_id"
    ).fetchone()[0]
    _fails(
        db,
        psycopg.errors.CheckViolation,
        "INSERT INTO run_steps (run_id, seq, turn, kind, payload) VALUES (%s, 1, 1, 'tool_call', '{}')",
        [run_id],
    )
    db.execute("INSERT INTO run_steps (run_id, seq, turn, kind, payload) VALUES (%s, 1, 1, 'model_turn', '[]')", [run_id])
    _fails(
        db,
        psycopg.errors.UniqueViolation,
        "INSERT INTO run_steps (run_id, seq, turn, kind, payload) VALUES (%s, 1, 1, 'model_turn', '[]')",
        [run_id],
    )


def test_0004_allows_waiting_and_its_downgrade_turns_waiting_into_failed(at_head_afterwards):
    url, cfg = at_head_afterwards, alembic_config(at_head_afterwards, TEST_DIMS)
    with connect(url) as conn:
        run_id = conn.execute(
            "INSERT INTO runs (task, model, agent_prompt_version, status) "
            "VALUES ('flow L1', 'm', 'flow-v2', 'waiting') RETURNING run_id"
        ).fetchone()[0]
    try:
        command.downgrade(cfg, "0003")
        with connect(url) as conn:
            assert conn.execute("SELECT status FROM runs WHERE run_id = %s", [run_id]).fetchone()[0] == "failed"
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute("UPDATE runs SET status = 'waiting' WHERE run_id = %s", [run_id])
    finally:
        with connect(url) as conn:
            conn.execute("DELETE FROM runs WHERE run_id = %s", [run_id])
