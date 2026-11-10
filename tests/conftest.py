"""Shared fixtures. `db`: a connection to the local TEST database, rolled back after each test.

The test database is migrated to head once per session (Alembic), so tests see the real schema.
"""

import os

import psycopg
import pytest

from research_assistant.db import connect, migrate

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
    """Real schema, rolled back after the test."""
    conn = connect(test_db_url)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
