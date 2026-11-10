"""Tests for rrf_merge, the function YOU write. Pure Python: no DB, no API."""

from datetime import date

import pytest

from research_assistant.retrieval import rrf_merge
from research_assistant.schemas import Chunk


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, source_url=f"https://x.example/{cid}", fetched_at=date(2026, 1, 1), title="t", text=cid)


def ranked(*ids: str, scores: list[float] | None = None):
    """A ranked list like keyword_search/vector_search return, best first."""
    scores = scores if scores is not None else [1.0 - 0.1 * i for i in range(len(ids))]
    return [(_chunk(cid), s) for cid, s in zip(ids, scores, strict=True)]


def ids(results):
    return [chunk.chunk_id for chunk, _ in results]


def test_single_list_keeps_its_order():
    assert ids(rrf_merge([ranked("a", "b", "c")])) == ["a", "b", "c"]


def test_fused_score_is_sum_of_reciprocal_ranks():
    # list 1: a(1) b(2)   list 2: b(1) c(2)
    out = {c.chunk_id: s for c, s in rrf_merge([ranked("a", "b"), ranked("b", "c")], k=60)}
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["c"] == pytest.approx(1 / 62)


def test_output_is_sorted_best_first():
    assert ids(rrf_merge([ranked("a", "b"), ranked("b", "c")])) == ["b", "a", "c"]


def test_agreement_beats_a_single_first_place():
    # "both" is only #2 in each list, but #2 + #2 > one #1.
    assert ids(rrf_merge([ranked("x", "both"), ranked("y", "both")]))[0] == "both"


def test_raw_scores_are_ignored():
    small = rrf_merge([ranked("a", "b", scores=[0.2, 0.1]), ranked("b", "a", scores=[0.9, 0.8])])
    huge = rrf_merge([ranked("a", "b", scores=[2000, 1000]), ranked("b", "a", scores=[0.9, 0.8])])
    assert small == huge


def test_ties_broken_by_chunk_id():
    # b is #1 in list 1, a is #1 in list 2: identical fused scores.
    assert ids(rrf_merge([ranked("b"), ranked("a")])) == ["a", "b"]


def test_top_k_truncates():
    assert ids(rrf_merge([ranked("a", "b", "c")], top_k=2)) == ["a", "b"]
    assert len(rrf_merge([ranked("a", "b", "c")], top_k=None)) == 3


def test_k_controls_the_formula():
    out = rrf_merge([ranked("a", "b")], k=0)
    assert [s for _, s in out] == pytest.approx([1.0, 0.5])


def test_empty_inputs():
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []
    assert ids(rrf_merge([[], ranked("a")])) == ["a"]


def test_returns_the_chunk_with_its_source():
    [(chunk, _)] = rrf_merge([ranked("a"), ranked("a")])
    assert chunk.source_url == "https://x.example/a"
