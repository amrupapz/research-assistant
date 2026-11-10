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
