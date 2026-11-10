"""Integration tests against a real local Postgres (stemming/ranking can't be mocked).

Uses a separate DB (TEST_DATABASE_URL, default research_assistant_test). Each test runs in a
transaction that is rolled back, so nothing persists. Skipped if Postgres isn't reachable.
"""

from datetime import date
from pathlib import Path

import pytest

from research_assistant.db import embed_missing, init_schema, sync_chunks
from research_assistant.ingest import load_chunks
from research_assistant.retrieval import hybrid_search, keyword_search, vector_search
from research_assistant.schemas import Chunk

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def pg(db):
    """Schema + the real data/ chunks (no embeddings)."""
    init_schema(db, 1536)
    sync_chunks(db, load_chunks(DATA))
    return db


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


def test_any_mode_finds_full_questions(pg):
    results = keyword_search(pg, "How much did Anthropic raise in its latest funding round?", match="any")
    assert _ids(results)[0] == "anthropic-funding#000"


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


# ---------- vector search with hand-made 3-d "embeddings" (no API) ----------

AXES = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "car": [0.0, 0.0, 1.0]}


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Each text points along the axis of the first animal/vehicle word it contains."""
    out = []
    for t in texts:
        if "kitten" in t:
            out.append([0.9, 0.3, 0.0])  # mostly cat, a bit dog
        else:
            out.append(next(v for word, v in AXES.items() if word in t))
    return out


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, source_url="https://x.example", fetched_at=date(2026, 1, 1), title="t", text=text)


@pytest.fixture
def pets(db):
    init_schema(db, 3)
    sync_chunks(db, [_chunk("c#0", "a cat"), _chunk("d#0", "a dog"), _chunk("v#0", "a car")])
    assert embed_missing(db, fake_embed) == 3
    return db


def test_vector_search_ranks_by_meaning(pets):
    results = vector_search(pets, "kitten", k=3, embed=fake_embed)
    assert _ids(results) == ["c#0", "d#0", "v#0"]
    scores = [s for _, s in results]
    assert scores[0] > scores[1] > scores[2]
    assert scores[2] == pytest.approx(0.0)  # orthogonal -> cosine similarity 0


def test_embed_missing_only_embeds_new_or_changed(pets):
    assert embed_missing(pets, fake_embed) == 0  # nothing new

    calls = []

    def spy(texts):
        calls.append(texts)
        return fake_embed(texts)

    sync_chunks(pets, [_chunk("c#0", "a cat"), _chunk("d#0", "a different dog"), _chunk("v#0", "a car")])
    assert embed_missing(pets, spy) == 1
    assert calls == [["t\n\na different dog"]]  # title + text, only the changed chunk


def test_hybrid_fuses_keyword_and_vector(pets):
    # keyword(any): 'kitten' | 'car' -> only v#0.   vector("kitten car" contains "kitten"): c, d, v.
    # RRF: v = 1/61 + 1/63 (both lists) > c = 1/61 > d = 1/62
    results = hybrid_search(pets, "kitten car", k=3, embed=fake_embed)
    assert _ids(results) == ["v#0", "c#0", "d#0"]
