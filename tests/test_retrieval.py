"""Unit tests with a fake connection: no Postgres needed."""

from datetime import date
from unittest.mock import MagicMock

from research_assistant.retrieval import keyword_search


def test_keyword_search_maps_rows_to_chunks_with_scores():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("intel-leadership#000", "https://intel.example", date(2026, 11, 7), "Intel Leadership", "Lip-Bu Tan…", 0.225),
    ]
    [(chunk, score)] = keyword_search(conn, "Lip-Bu Tan", k=3)
    assert chunk.chunk_id == "intel-leadership#000"
    assert chunk.fetched_at == date(2026, 11, 7)
    assert score == 0.225


def test_keyword_search_passes_query_as_a_parameter_not_in_sql():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    evil = "x'; DROP TABLE chunks; --"
    assert keyword_search(conn, evil, k=7) == []
    sql, params = conn.execute.call_args.args
    assert evil not in sql
    assert params == {"q": evil, "k": 7}


def test_vector_search_embeds_query_once_and_passes_vector_as_parameter():
    from research_assistant.retrieval import vector_search

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    calls = []

    def fake_embed(texts):
        calls.append(texts)
        return [[0.5, -1.0, 2.0]]

    assert vector_search(conn, "headcount", k=4, embed=fake_embed) == []
    assert calls == [["headcount"]]
    _, params = conn.execute.call_args.args
    assert params == {"v": "[0.5,-1.0,2.0]", "k": 4}


def test_to_pgvector_format():
    from research_assistant.db import to_pgvector

    assert to_pgvector([1, 0.25, -3e-05]) == "[1.0,0.25,-3e-05]"
