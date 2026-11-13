"""Shared fixtures. `db`: a connection to the local TEST database, rolled back after each test.

The test database is migrated to head once per session (Alembic), so tests see the real schema.
"""

import os

import psycopg
import pytest
from opentelemetry.trace import StatusCode

from research_assistant import tracing
from research_assistant.db import connect, migrate, seed_leads

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/research_assistant_test")
TEST_DIMS = 1536


def db_unreachable(reason: str) -> None:
    """Locally: skip (no Postgres on this laptop is fine). In CI (GitHub sets CI=true): FAIL. A skipped
    DB suite shows green with ~200 tests never run, which is worse than red (D7 M5)."""
    if os.environ.get("CI", "").lower() == "true":
        pytest.fail(f"CI=true but {reason}: the DB tests must run in CI", pytrace=False)
    pytest.skip(f"{reason}; run: createdb research_assistant_test")


@pytest.fixture(scope="session")
def test_db_url():
    """The test database URL, migrated to head. Skips if Postgres isn't reachable (fails in CI)."""
    try:
        connect(TEST_DB).close()
    except psycopg.OperationalError as e:
        db_unreachable(f"Postgres not reachable ({e.__class__.__name__})")
    migrate(TEST_DB, dims=TEST_DIMS)
    return TEST_DB


@pytest.fixture
def db(test_db_url):
    """Real schema, rolled back after the test.

    The SELECT opens the outer transaction NOW, so every `with conn.transaction():` in the code
    under test is a savepoint inside it, and nothing ever commits to the test DB.
    """
    conn = connect(test_db_url)
    conn.execute("SELECT 1")
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


AGENT_TABLES = "runs, run_steps, drafts, approvals, sends, qualifications, research_findings, finding_citations, replies"


@pytest.fixture
def state_db(db):
    """Empty agent-state tables + the 4 seed leads. All rolled back.

    RESTART IDENTITY: sequences are NOT transactional (a rolled-back INSERT still uses up its id),
    so without it the first draft in a test could be D37. TRUNCATE ... RESTART IDENTITY is the
    exception: inside a transaction, the restart rolls back too.
    """
    db.execute(f"TRUNCATE {AGENT_TABLES} RESTART IDENTITY")
    seed_leads(db)
    return db


# --- tracing (D6 M1): tests NEVER send traces ---------------------------------------------------


@pytest.fixture(autouse=True)
def tracing_off():
    """Every test starts and ends with tracing off, whatever .env says. Tests that check spans
    opt in with the `spans` fixture (real SDK, in-memory exporter: nothing leaves the process)."""
    tracing.use(None)
    yield
    tracing.use(None)


@pytest.fixture(scope="session")
def _memory_langfuse():
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    lf = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",  # nothing listens there; the exporter below replaces the network one
        span_exporter=exporter,
        mask=tracing.mask,
    )
    yield lf, exporter
    lf.shutdown()


@pytest.fixture
def spans(_memory_langfuse):
    """Turn tracing on for this test; call the result to get the finished observations as dicts:
    {name, type, parent (name or None), trace_id, level, metadata, attrs}."""
    lf, exporter = _memory_langfuse
    exporter.clear()
    tracing.use(lf)

    def finished() -> list[dict]:
        lf.flush()
        raw = exporter.get_finished_spans()
        names = {s.context.span_id: s.name for s in raw}
        out = []
        for s in raw:
            a = dict(s.attributes)
            prefix = "langfuse.observation.metadata."
            out.append({
                "name": s.name,
                "type": a.get("langfuse.observation.type"),
                "parent": names.get(s.parent.span_id) if s.parent else None,
                "trace_id": format(s.context.trace_id, "032x"),
                # Like the Langfuse server: no explicit level + OpenTelemetry status ERROR -> ERROR.
                # (Checking only the attribute missed every paused approve showing ERROR live.)
                "level": a.get(
                    "langfuse.observation.level", "ERROR" if s.status.status_code == StatusCode.ERROR else "DEFAULT"
                ),
                "metadata": {k.removeprefix(prefix): v for k, v in a.items() if k.startswith(prefix)},
                "attrs": a,
            })
        return out

    return finished
