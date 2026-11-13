"""D7 M4: `python -m research_assistant.db --schema-only` (Docker's migrate service) never ingests or embeds."""

import pytest

from research_assistant import db, ingest, llm


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(db, "setup_schema", lambda url, dims: seen.append("schema"))
    monkeypatch.setattr(ingest, "load_chunks", lambda: seen.append("ingest") or [])
    monkeypatch.setattr(llm, "embed", lambda texts: seen.append("embed") or [])

    class Settings:
        database_url = "postgresql://unused"
        embedding_dims = 1536

    monkeypatch.setattr("research_assistant.config.get_settings", lambda: Settings)
    return seen


def test_schema_only_migrates_and_stops(calls):
    db.main(["--schema-only"])
    assert calls == ["schema"]  # no ingest, no paid embedding calls on container start


def test_unknown_flag_is_an_error_not_a_full_ingest(calls):
    with pytest.raises(SystemExit):
        db.main(["--schema"])
    assert calls == []
