"""Shared fixtures. `db`: a connection to the local TEST database, rolled back after each test."""

import os

import psycopg
import pytest

from research_assistant.db import connect

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/research_assistant_test")


@pytest.fixture
def db():
    """Empty schema, rolled back after the test. Skipped if Postgres isn't reachable."""
    try:
        conn = connect(TEST_DB)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable ({e.__class__.__name__}); run: createdb research_assistant_test")
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
