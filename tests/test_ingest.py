from datetime import date
from itertools import pairwise
from pathlib import Path

import pytest

from research_assistant.ingest import IngestError, chunk_text, load_chunks, parse_page

HEADER = "---\nurl: https://example.com/a\nfetched_at: 2026-11-01\ntitle: A page\n---\n"


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_parse_page_reads_header_and_body():
    page = parse_page(HEADER + "Hello world.")
    assert page.source_url == "https://example.com/a"
    assert page.fetched_at == date(2026, 11, 1)
    assert page.title == "A page"
    assert page.text == "Hello world."
    assert page.synthetic is False


@pytest.mark.parametrize(
    "raw",
    [
        "No header at all.",
        "---\nurl: https://example.com\ntitle: x\n---\nbody",  # no fetched_at
        "---\nfetched_at: 2026-01-01\ntitle: x\n---\nbody",  # no url
        "---\nurl: https://example.com\nfetched_at: 2026-01-01\ntitle: x\nbody",  # header never closed
    ],
)
def test_page_without_source_header_is_rejected(raw):
    with pytest.raises(IngestError):
        parse_page(raw)


def test_consecutive_chunks_share_exactly_overlap_words():
    chunks = chunk_text(_words(500), size=200, overlap=40)
    for prev, nxt in pairwise(chunks):
        assert prev.split()[-40:] == nxt.split()[:40]


def test_chunks_cover_every_word_and_respect_size():
    chunks = chunk_text(_words(500), size=200, overlap=40)
    assert all(len(c.split()) <= 200 for c in chunks)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w499"


def test_short_text_is_one_chunk_and_empty_is_none():
    assert chunk_text(_words(50), size=200, overlap=40) == [_words(50)]
    assert chunk_text("   ", size=200, overlap=40) == []


def test_chunking_keeps_line_breaks():
    text = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert chunk_text(text, size=200, overlap=40) == [text]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)


def test_load_chunks_ids_are_stable_and_carry_source(tmp_path: Path):
    (tmp_path / "b-page.md").write_text(HEADER + _words(300))
    (tmp_path / "a-page.md").write_text(HEADER + "short")

    first = load_chunks(tmp_path, size=200, overlap=40)
    second = load_chunks(tmp_path, size=200, overlap=40)

    assert [c.chunk_id for c in first] == ["a-page#000", "b-page#000", "b-page#001"]
    assert first == second
    assert all(c.source_url and c.fetched_at for c in first)


def test_real_data_dir_ingests():
    chunks = load_chunks(Path(__file__).parent.parent / "data")
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.chunk_id.split("#")[0] for c in chunks} >= {"anthropic-funding", "intel-leadership-2024"}
