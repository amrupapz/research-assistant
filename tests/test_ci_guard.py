"""D7 M5: in CI an unreachable test DB must FAIL the run, not skip ~200 tests into a green check."""

import pytest
from _pytest.outcomes import Failed, Skipped
from conftest import db_unreachable


def test_locally_it_skips(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(Skipped):
        db_unreachable("Postgres not reachable")


def test_in_ci_it_fails(monkeypatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(Failed, match="must run in CI"):
        db_unreachable("Postgres not reachable")
