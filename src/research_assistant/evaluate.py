"""Pure scoring for evals: retrieval and answers are graded SEPARATELY. No DB, no API here."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from research_assistant.schemas import Chunk, ResearchAnswer

Expect = Literal["answer", "conflict", "abstain"]


class EvalCase(BaseModel):
    id: str
    question: str
    kind: str
    expect: Expect
    gold_sources: list[str] = []  # data/ filenames, e.g. "figma-about.md"
    gold_facts: list[str] = []  # strings that must appear, e.g. "2,140"


def load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def normalize(text: str) -> str:
    """Lowercase and drop thousands separators, so '2,140' matches '2140'."""
    return text.lower().replace(",", "")


def source_of(chunk_id: str) -> str:
    """'figma-about#002' -> 'figma-about' (the data/ file stem)."""
    return chunk_id.split("#")[0]


def _gold_stems(case: EvalCase) -> set[str]:
    return {Path(s).stem for s in case.gold_sources}


def _contains(haystack: str, fact: str) -> bool:
    return normalize(fact) in normalize(haystack)


def retrieval_scores(case: EvalCase, chunks: list[Chunk]) -> dict:
    """Did the top-k chunks contain the evidence?

    file_recall: share of gold files with at least one chunk retrieved (right page?)
    fact_recall: share of gold facts appearing in some retrieved chunk's text (right chunk?)
    hit:         both are 1.0
    """
    gold = _gold_stems(case)
    got = {source_of(c.chunk_id) for c in chunks}
    file_recall = len(gold & got) / len(gold) if gold else 1.0
    found = [f for f in case.gold_facts if any(_contains(c.text, f) for c in chunks)]
    fact_recall = len(found) / len(case.gold_facts) if case.gold_facts else 1.0
    return {"file_recall": file_recall, "fact_recall": fact_recall, "hit": file_recall == 1.0 and fact_recall == 1.0}


def grade_answer(case: EvalCase, result: ResearchAnswer | None) -> dict[str, bool]:
    """Named pass/fail checks on the structured answer (not exact wording). result=None means ask() raised."""
    checks: dict[str, bool] = {"grounded": result is not None}
    if result is None:
        checks["pass"] = False
        return checks

    cited = {source_of(c.chunk_id) for c in result.citations}
    gold = _gold_stems(case)
    if case.expect == "answer":
        checks["has_facts"] = all(_contains(result.answer, f) for f in case.gold_facts)
        checks["cites_gold"] = bool(cited & gold)
    elif case.expect == "conflict":
        checks["has_facts"] = all(_contains(result.answer, f) for f in case.gold_facts)
        checks["cites_all_gold"] = gold <= cited
        checks["flags_needs_sources"] = result.needs_sources
    else:  # abstain
        checks["no_citations"] = not result.citations
        checks["flags_needs_sources"] = result.needs_sources
    checks["pass"] = all(checks.values())
    return checks
