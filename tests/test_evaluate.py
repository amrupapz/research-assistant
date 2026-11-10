"""Scoring functions for evals, plus sanity checks on the eval set itself. No DB, no API."""

from datetime import date
from pathlib import Path

import pytest

from research_assistant.evaluate import (
    EvalCase,
    grade_answer,
    load_cases,
    retrieval_scores,
)
from research_assistant.ingest import load_chunks
from research_assistant.schemas import Chunk, Citation, ResearchAnswer

ROOT = Path(__file__).parent.parent
FIGMA = EvalCase(
    id="t", question="q", kind="k", expect="conflict",
    gold_sources=["figma-about.md", "figma-company-profile.md"], gold_facts=["2,140", "1,870"],
)
INTEL = EvalCase(id="t", question="q", kind="k", expect="answer", gold_sources=["intel.md"], gold_facts=["Lip-Bu Tan"])
ABSTAIN = EvalCase(id="t", question="q", kind="k", expect="abstain")


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, source_url="https://x", fetched_at=date(2026, 1, 1), title="t", text=text)


def _answer(text: str, cited: list[str], needs_sources: bool) -> ResearchAnswer:
    return ResearchAnswer(
        answer=text, confidence="high", assumptions=[], needs_sources=needs_sources,
        citations=[Citation(chunk_id=c, source_url="https://x") for c in cited],
    )


# ---------- retrieval_scores ----------


def test_retrieval_hit_needs_right_files_and_right_facts():
    chunks = [_chunk("figma-about#000", "**2,140** Figmates"), _chunk("figma-company-profile#000", "Estimated: 1,870")]
    assert retrieval_scores(FIGMA, chunks) == {"file_recall": 1.0, "fact_recall": 1.0, "hit": True}


def test_right_file_wrong_chunk_is_not_a_hit():
    # the M1 check-question gap: file-level labels alone would call this a hit
    chunks = [_chunk("figma-about#001", "offices"), _chunk("figma-company-profile#001", "| 9% |")]
    s = retrieval_scores(FIGMA, chunks)
    assert (s["file_recall"], s["fact_recall"], s["hit"]) == (1.0, 0.0, False)


def test_partial_recall_and_comma_normalization():
    s = retrieval_scores(FIGMA, [_chunk("figma-about#000", "2140 people")])
    assert (s["file_recall"], s["fact_recall"]) == (0.5, 0.5)


# ---------- grade_answer ----------


def test_answer_passes_with_fact_and_gold_citation():
    checks = grade_answer(INTEL, _answer("Lip-Bu Tan [intel#000]", ["intel#000"], False))
    assert checks == {"grounded": True, "has_facts": True, "cites_gold": True, "pass": True}


def test_answer_with_wrong_fact_fails():
    checks = grade_answer(INTEL, _answer("Pat Gelsinger [intel-2024#000]", ["intel-2024#000"], False))
    assert not checks["has_facts"] and not checks["cites_gold"] and not checks["pass"]


def test_conflict_requires_both_values_both_sources_and_flag():
    good = _answer("2,140 [figma-about#000] vs 1,870 [figma-company-profile#000]",
                   ["figma-about#000", "figma-company-profile#000"], True)
    assert grade_answer(FIGMA, good)["pass"]

    # the q07b failure from M5: one source, confident, no flag
    one_sided = _answer("Figma has 2,140 people [figma-about#000]", ["figma-about#000"], False)
    checks = grade_answer(FIGMA, one_sided)
    assert checks == {"grounded": True, "has_facts": False, "cites_all_gold": False,
                      "flags_needs_sources": False, "pass": False}


def test_abstain_checks():
    assert grade_answer(ABSTAIN, _answer("The provided sources don't cover this.", [], True))["pass"]
    assert not grade_answer(ABSTAIN, _answer("Stripe sells payments [x#000]", ["x#000"], True))["pass"]


def test_raised_error_is_a_failed_grounding():
    assert grade_answer(INTEL, None) == {"grounded": False, "pass": False}


# ---------- the eval set itself ----------


def test_eval_set_is_consistent_with_data():
    """Every gold fact must really be in its gold files, or the eval is broken, not the system."""
    cases = load_cases(ROOT / "evals" / "questions.jsonl")
    chunks = load_chunks(ROOT / "data")
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        if case.expect == "abstain":
            assert not case.gold_sources and not case.gold_facts, case.id
            continue
        gold_chunks = [c for c in chunks if c.chunk_id.split("#")[0] in {Path(s).stem for s in case.gold_sources}]
        assert retrieval_scores(case, gold_chunks)["hit"], f"{case.id}: gold facts not found in gold files"


@pytest.mark.parametrize("expect", ["answer", "conflict", "abstain"])
def test_every_expect_type_is_used(expect):
    assert any(c.expect == expect for c in load_cases(ROOT / "evals" / "questions.jsonl"))
