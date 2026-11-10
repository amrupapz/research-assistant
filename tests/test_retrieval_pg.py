"""Integration tests against a real local Postgres (stemming/ranking can't be mocked).

Uses a separate DB (TEST_DATABASE_URL, default research_assistant_test). Each test runs in a
transaction that is rolled back, so nothing persists. Skipped if Postgres isn't reachable.
"""

import os
from datetime import date
from pathlib import Path

import psycopg
import pytest

from research_assistant.db import connect, init_schema, sync_chunks
from research_assistant.ingest import load_chunks
from research_assistant.retrieval import keyword_search
from research_assistant.schemas import Chunk

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/research_assistant_test")
DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def pg():
    try:
        conn = connect(TEST_DB)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable ({e.__class__.__name__}); run: createdb research_assistant_test")
    try:
        init_schema(conn)
        sync_chunks(conn, load_chunks(DATA))
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _ids(results):
    return [chunk.chunk_id for chunk, _ in results]


@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        ("Lip-Bu Tan", "intel-leadership#000"),
        ("Series H", "anthropic-funding#000"),
        ("Agent Runtime", "vercel-changelog#000"),
        ("Pat Gelsinger", "intel-leadership-2024#000"),
    ],
)
def test_exact_names_rank_the_right_chunk_first(pg, query, expected_top):
    assert _ids(keyword_search(pg, query))[0] == expected_top


def test_stemming_matches_other_word_forms(pg):
    # the page says "raises"/"raised"; both stem to 'rais'
    assert _ids(keyword_search(pg, "raising"))[0] == "anthropic-funding#000"


def test_full_question_finds_nothing_because_words_are_anded(pg):
    # Known limitation of keyword search, kept as a test so we notice if it changes.
    assert keyword_search(pg, "How much did Anthropic raise in its latest funding round?") == []


def test_respects_k(pg):
    assert len(keyword_search(pg, "Intel", k=1)) == 1


def test_sync_is_idempotent_and_prunes_stale_chunks(pg):
    stale = Chunk(
        chunk_id="gone#000", source_url="https://x.example", fetched_at=date(2020, 1, 1), title="t", text="t"
    )
    chunks = load_chunks(DATA)
    sync_chunks(pg, [*chunks, stale])
    assert sync_chunks(pg, chunks) == 1
    assert sync_chunks(pg, chunks) == 0
    assert pg.execute("SELECT count(*) FROM chunks").fetchone()[0] == len(chunks)
