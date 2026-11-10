"""Shared fixtures. `db`: a connection to the local TEST database, rolled back after each test.

The test database is migrated to head once per session (Alembic), so tests see the real schema.
"""

import os

import psycopg
import pytest

from research_assistant.db import connect, migrate, seed_leads

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/research_assistant_test")
TEST_DIMS = 1536


@pytest.fixture(scope="session")
def test_db_url():
    """The test database URL, migrated to head. Skips if Postgres isn't reachable."""
    try:
        connect(TEST_DB).close()
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable ({e.__class__.__name__}); run: createdb research_assistant_test")
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
