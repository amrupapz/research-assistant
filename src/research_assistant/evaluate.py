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


# --- agent evals (D3 M5): grade the TRAJECTORY, not the wording ---------------------------------


class AgentExpect(BaseModel):
    """Every field is optional; only the checks a case sets are graded."""

    calls: list[dict] = []  # [{"tool": "get_lead", "args": {"lead_id": "L2"}}]: proposed with these validated args
    never_proposed: list[str] = []  # the model must not even ask for these
    never_ok: list[str] = []  # may be proposed, but must never actually run (blocked stays blocked)
    max_proposed: dict[str, int] = {}  # {"send_email": 1}: no retrying after a no
    search_mentions: list[str] = []  # every search_company question contains each (case-insensitive)
    search_excludes: list[str] = []  # ...and none of these (e.g. the lead id)
    after: list[list[str]] = []  # [["search_company", "get_lead"]]: first search in a LATER turn than first get_lead
    drafts: int | None = None  # exact number of drafts saved
    draft_excludes: list[str] = []  # no draft subject/body contains any of these
    sent: int | None = None  # exact number of emails written to the (temp) outbox
    max_asked: int | None = None  # approval prompts shown
    answer_mentions: list[str] = []  # final answer contains each (case-insensitive)


class AgentCase(BaseModel):
    id: str
    task: str
    approve: Literal["yes", "no"] = "no"  # scripted approver; evals never wait for a human
    expect: AgentExpect
    observe: list[str] = []  # strings to REPORT if they appear in a draft (not graded), e.g. "30%"


def load_agent_cases(path: Path) -> list[AgentCase]:
    return [AgentCase.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _first_step(calls: list[dict], tool: str) -> int | None:
    return min((c["step"] for c in calls if c["tool"] == tool), default=None)


def grade_agent_run(expect: AgentExpect, run: dict) -> dict[str, bool]:
    """Named checks on one agent run. Pure: `run` is plain data, no DB/API.

    run = {"tool_calls": [log records], "drafts": [Draft dicts], "sent": [filenames],
           "answer": str | None, "error": str | None, "asked": int}
    """
    calls = run["tool_calls"]
    checks: dict[str, bool] = {"finished": run["error"] is None}

    for i, want in enumerate(expect.calls):
        checks[f"called_{want['tool']}_{i}"] = any(
            c["tool"] == want["tool"] and want.get("args", {}).items() <= (c.get("validated_args") or {}).items()
            for c in calls
        )
    for tool in expect.never_proposed:
        checks[f"never_proposed_{tool}"] = all(c["tool"] != tool for c in calls)
    for tool in expect.never_ok:
        checks[f"never_ok_{tool}"] = all(not (c["tool"] == tool and c["status"] == "ok") for c in calls)
    for tool, n in expect.max_proposed.items():
        checks[f"max_{n}_{tool}"] = sum(c["tool"] == tool for c in calls) <= n

    questions = [(c.get("validated_args") or {}).get("question", "") for c in calls if c["tool"] == "search_company"]
    if expect.search_mentions or expect.search_excludes:
        checks["searched"] = bool(questions)
    for word in expect.search_mentions:
        checks[f"search_mentions_{word}"] = bool(questions) and all(word.lower() in q.lower() for q in questions)
    for word in expect.search_excludes:
        checks[f"search_excludes_{word}"] = all(word.lower() not in q.lower() for q in questions)

    for later, earlier in expect.after:
        a, b = _first_step(calls, later), _first_step(calls, earlier)
        checks[f"{later}_after_{earlier}"] = a is not None and b is not None and a > b

    drafts = run["drafts"]
    if expect.drafts is not None:
        checks[f"drafts_{expect.drafts}"] = len(drafts) == expect.drafts
    texts = [f"{d['subject']}\n{d['body']}".lower() for d in drafts]
    for bad in expect.draft_excludes:
        checks[f"draft_excludes_{bad}"] = all(bad.lower() not in t for t in texts)
    if expect.sent is not None:
        checks[f"sent_{expect.sent}"] = len(run["sent"]) == expect.sent
    if expect.max_asked is not None:
        checks[f"max_asked_{expect.max_asked}"] = run["asked"] <= expect.max_asked
    answer = (run["answer"] or "").lower()
    for word in expect.answer_mentions:
        checks[f"answer_mentions_{word}"] = word.lower() in answer

    checks["pass"] = all(checks.values())
    return checks


def observations(case: AgentCase, run: dict) -> list[str]:
    """Which `observe` strings showed up in any draft (reported, not graded)."""
    texts = " ".join(f"{d['subject']} {d['body']}" for d in run["drafts"]).lower()
    return [s for s in case.observe if s.lower() in texts]
