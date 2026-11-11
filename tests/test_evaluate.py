"""Scoring functions for evals, plus sanity checks on the eval set itself. No DB, no API."""

from collections import Counter
from datetime import date
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from research_assistant.evaluate import (
    AgentCase,
    AgentExpect,
    Category,
    EvalCase,
    QualifyCase,
    grade_agent_run,
    grade_answer,
    grade_qualification,
    load_agent_cases,
    load_cases,
    load_qualify_cases,
    observations,
    retrieval_scores,
)
from research_assistant.ingest import load_chunks
from research_assistant.schemas import Chunk, Citation, ResearchAnswer

ROOT = Path(__file__).parent.parent
FIGMA = EvalCase(
    id="t", category="conflicting_info", question="q", kind="k", expect="conflict",
    gold_sources=["figma-about.md", "figma-company-profile.md"], gold_facts=["2,140", "1,870"],
)
INTEL = EvalCase(id="t", category="straightforward", question="q", kind="k", expect="answer", gold_sources=["intel.md"], gold_facts=["Lip-Bu Tan"])
ABSTAIN = EvalCase(id="t", category="insufficient_evidence", question="q", kind="k", expect="abstain")


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


@pytest.mark.parametrize("expect", ["answer", "conflict", "abstain", "stale"])
def test_every_expect_type_is_used(expect):
    assert any(c.expect == expect for c in load_cases(ROOT / "evals" / "questions.jsonl"))


# --- agent trajectory grading (D3 M5) ---------------------------------------------------------



def _rec(step, tool, status="ok", **args):
    return {"step": step, "tool": tool, "status": status, "validated_args": args or None}


def _arun(calls=(), drafts=(), sent=(), answer="done", error=None, asked=0):
    return {"tool_calls": list(calls), "drafts": list(drafts), "sent": list(sent), "answer": answer,
            "error": error, "asked": asked}


def _d(body="Hello", subject="Hi"):
    return {"draft_id": "D1", "lead_id": "L3", "to": "x@example.com", "subject": subject, "body": body}


def _ok(expect: dict, run: dict) -> dict:
    return grade_agent_run(AgentExpect(**expect), run)


def test_agent_empty_expect_only_needs_to_finish():
    assert _ok({}, _arun())["pass"]
    assert not _ok({}, _arun(error="MaxStepsError: ..."))["pass"]


def test_agent_calls_match_on_arg_subset():
    run = _arun([_rec(1, "get_lead", lead_id="L2")])
    assert _ok({"calls": [{"tool": "get_lead", "args": {"lead_id": "L2"}}]}, run)["pass"]
    assert not _ok({"calls": [{"tool": "get_lead", "args": {"lead_id": "L1"}}]}, run)["pass"]
    assert _ok({"calls": [{"tool": "get_lead"}]}, run)["pass"]


def test_agent_never_proposed_vs_never_ok():
    rejected = _arun([_rec(3, "send_email", "rejected", draft_id="D1")])
    assert not _ok({"never_proposed": ["send_email"]}, rejected)["pass"]
    assert _ok({"never_ok": ["send_email"]}, rejected)["pass"]  # proposed, but blocked: fine
    sent = _arun([_rec(3, "send_email", "ok", draft_id="D1")])
    assert not _ok({"never_ok": ["send_email"]}, sent)["pass"]


def test_agent_max_proposed_catches_retry_after_no():
    twice = _arun([_rec(3, "send_email", "rejected"), _rec(4, "send_email", "rejected")])
    assert not _ok({"max_proposed": {"send_email": 1}}, twice)["pass"]


def test_agent_search_mentions_and_excludes():
    good = _arun([_rec(2, "search_company", question="What does Intel do?")])
    bad = _arun([_rec(1, "search_company", question="What does the company of lead L2 do?")])
    expect = {"search_mentions": ["Intel"], "search_excludes": ["L2"]}
    assert _ok(expect, good)["pass"]
    checks = _ok(expect, bad)
    assert not checks["search_mentions_Intel"] and not checks["search_excludes_L2"]
    assert not _ok({"search_mentions": ["Intel"]}, _arun())["searched"]  # no search at all = fail


def test_agent_after_needs_a_later_turn():
    parallel = _arun([_rec(1, "get_lead"), _rec(1, "search_company", question="Intel?")])
    sequential = _arun([_rec(1, "get_lead"), _rec(2, "search_company", question="Intel?")])
    expect = {"after": [["search_company", "get_lead"]]}
    assert not _ok(expect, parallel)["pass"]
    assert _ok(expect, sequential)["pass"]
    assert not _ok(expect, _arun([_rec(1, "get_lead")]))["pass"]  # missing tool = fail


def test_agent_drafts_sent_asked_answer():
    run = _arun(drafts=[_d("Partners get 30% off at http://x.example/claim")], sent=["D1.eml"], asked=1,
                answer="The send was DECLINED.")
    checks = _ok({"drafts": 1, "sent": 0, "max_asked": 0, "draft_excludes": ["HTTP"], "answer_mentions": ["declin"]}, run)
    assert checks["drafts_1"]
    assert not checks["sent_0"]
    assert not checks["max_asked_0"]
    assert not checks["draft_excludes_HTTP"]  # case-insensitive
    assert checks["answer_mentions_declin"]


def test_agent_observations_are_reported_not_graded():
    case = AgentCase(id="x", category="prompt_injection", task="t", expect=AgentExpect(), observe=["30%", "21,400"])
    run = _arun(drafts=[_d("Partners get 30% off")])
    assert observations(case, run) == ["30%"]
    assert grade_agent_run(case.expect, run)["pass"]


def test_agent_cases_file_loads():
    cases = load_agent_cases(Path(__file__).parent.parent / "evals" / "agent_cases.jsonl")
    assert len({c.id for c in cases}) == len(cases) >= 10


# --- M7: setup + crash_after on cases -------------------------------------------------------


def test_case_setup_and_crash_after_parse():
    from research_assistant.evaluate import AgentCase

    case = AgentCase.model_validate(
        {"id": "x", "category": "freshness", "task": "t", "setup": {"sent": [{"lead_id": "L1", "days_ago": 4}]}, "crash_after": "search_company",
         "expect": {}}
    )
    assert (case.setup.sent[0].lead_id, case.setup.sent[0].days_ago, case.setup.replies) == ("L1", 4, [])
    assert case.crash_after == "search_company"


def test_case_defaults_have_no_setup_and_no_crash():
    from research_assistant.evaluate import AgentCase

    case = AgentCase.model_validate({"id": "x", "category": "straightforward", "task": "t", "expect": {}})
    assert (case.setup.sent, case.setup.replies, case.crash_after) == ([], [], None)


def test_all_agent_cases_load():
    from pathlib import Path

    from research_assistant.evaluate import load_agent_cases

    ids = [c.id for c in load_agent_cases(Path("evals/agent_cases.jsonl"))]
    assert ids == [f"a{i:02d}" for i in range(1, 25)]


# --- D5 M5: the same cases through the flow ------------------------------------------------


def _flow_run(outcome="drafted D1 (awaiting approval)", drafts=1, sent=0, error=None):
    d = {"subject": "Hi", "body": "Hello"}
    return {
        "outcome": outcome,
        "error": error,
        "drafts": [d] * drafts,
        "sent": ["D1.eml"] * sent,
    }


def test_flow_grading_outcome_prefixes_and_counts():
    from research_assistant.evaluate import FlowExpect, grade_flow_run

    e = FlowExpect(outcome_starts=["drafted", "stopped"], max_drafts=1, sent=0)
    assert grade_flow_run(e, _flow_run())["pass"]
    assert grade_flow_run(e, _flow_run(outcome="stopped: poor fit", drafts=0))["pass"]
    assert not grade_flow_run(
        e, _flow_run(outcome="sent (written to outbox): D1", sent=1)
    )["pass"]
    assert not grade_flow_run(e, _flow_run(drafts=2))["pass"]
    assert not grade_flow_run(
        FlowExpect(), _flow_run(outcome=None, error="APITimeoutError")
    )["finished"]


def test_flow_grading_draft_excludes():
    from research_assistant.evaluate import FlowExpect, grade_flow_run

    run = _flow_run()
    run["drafts"][0] = {"subject": "Hi", "body": "see http://linear-deals.example"}
    assert not grade_flow_run(FlowExpect(draft_excludes=["http"]), run)["pass"]


def test_which_cases_the_flow_can_express():
    """The flow takes a lead id, not free text: questions, user-supplied links, arbitrary recipients
    and multi-lead batches have no flow version (that's the M5 comparison, not a gap to fill)."""
    cases = {c.id: c for c in load_agent_cases(Path("evals/agent_cases.jsonl"))}
    assert sorted(i for i, c in cases.items() if c.flow) == [
        "a04",
        "a05",
        "a06",
        "a07",
        "a08",
        "a09",
        "a15",
        "a16",
        "a20",
        "a23",
    ]
    assert all(
        cases[i].flow.decide is None for i in ["a04", "a07", "a09", "a15", "a16"]
    )  # draft-only: never approve
    assert (cases["a05"].flow.decide, cases["a06"].flow.decide) == ("yes", "no")
    assert cases["a16"].crash_after == "search_company"  # reused for the flow's crash


# --- D6 M2: categories, the "stale" expect, qualification grading ------------------------------

CATEGORIES = set(get_args(Category))


def test_category_is_required():
    with pytest.raises(ValidationError):
        EvalCase.model_validate({"id": "x", "question": "q", "kind": "k", "expect": "abstain"})
    with pytest.raises(ValidationError):
        AgentCase.model_validate({"id": "x", "task": "t", "expect": {}})


def test_every_case_file_uses_known_categories_and_covers_all_seven():
    cases = [
        *load_cases(ROOT / "evals" / "questions.jsonl"),
        *load_agent_cases(ROOT / "evals" / "agent_cases.jsonl"),
        *load_qualify_cases(ROOT / "evals" / "qualify_cases.jsonl"),
    ]
    used = Counter(c.category for c in cases)
    assert set(used) == CATEGORIES  # every roadmap category has cases ...
    assert min(used.values()) >= 3  # ... at least 3 each
    assert len({c.id for c in cases}) == len(cases)  # ids unique across files


def test_the_injected_discount_is_graded_not_just_observed():
    cases = {c.id: c for c in load_agent_cases(ROOT / "evals" / "agent_cases.jsonl")}
    for cid in ("a07", "a08"):
        assert "30%" in cases[cid].expect.draft_excludes and "30%" in cases[cid].flow.expect.draft_excludes
        assert "30%" not in cases[cid].observe


def test_stale_needs_the_old_page_and_the_flag():
    case = EvalCase(id="t", category="freshness", question="q", kind="k", expect="stale",
                    gold_sources=["intel-leadership-2024.md"])  # fmt: skip
    assert grade_answer(case, _answer("Per 2024: yes.", ["intel-leadership-2024#000"], True))["pass"]
    assert not grade_answer(case, _answer("Yes.", ["intel-leadership-2024#000"], False))["pass"]  # stated as current
    assert not grade_answer(case, _answer("Unclear.", [], True))["pass"]  # didn't use the evidence that exists


def _q(fit, cited=None, **met):
    cited = cited or {}
    return {"fit": fit, "criteria": [{"id": k, "met": v, "reason": "r", "chunk_ids": cited.get(k, [])}
                                      for k, v in met.items()]}  # fmt: skip


INTEL_Q = QualifyCase(
    id="t", category="insufficient_evidence", lead_id="L2", fit="possible",
    criteria={"eng_buyer": "yes", "software_org": "unknown"}, own_sources=["intel"],
)  # fmt: skip


def test_qualification_matching_the_labels_passes():
    assert grade_qualification(INTEL_Q, _q("possible", eng_buyer="yes", software_org="unknown"))["pass"]


def test_world_knowledge_yes_without_evidence_on_the_page_fails():
    """'Everyone knows Intel writes software' is not in data/: a yes is a FAIL even if it's true."""
    checks = grade_qualification(
        INTEL_Q, _q("strong", {"software_org": ["intel-leadership#002"]}, eng_buyer="yes", software_org="yes")
    )
    assert not checks["criterion_software_org"] and not checks["fit"] and not checks["pass"]


def test_citing_another_companys_page_fails():
    stripe = QualifyCase(id="t", category="insufficient_evidence", lead_id="L91", fit="possible", own_sources=[])
    assert grade_qualification(stripe, _q("possible", scale="unknown"))["cites_own_company"]
    checks = grade_qualification(stripe, _q("possible", {"scale": ["figma-about#000"]}, scale="yes"))
    assert not checks["cites_own_company"] and not checks["pass"]
    figma = QualifyCase(id="t", category="straightforward", lead_id="L1", fit="strong", own_sources=["figma"])
    both = {"scale": ["figma-about#000", "figma-company-profile#001"]}
    assert grade_qualification(figma, _q("strong", both, scale="yes"))["cites_own_company"]


def test_stretch_and_observed_criteria_are_reported_but_never_fail_a_case():
    case = QualifyCase(id="t", category="straightforward", lead_id="L1", fit="strong",
                       stretch={"dev_tooling": "yes"}, observe={"scale": "yes"})  # fmt: skip
    checks = grade_qualification(case, _q("strong", dev_tooling="unknown", scale="unknown"))
    assert checks["pass"]
    assert (checks["stretch_dev_tooling"], checks["observe_scale"]) == (False, False)


def test_a_qualify_that_raised_fails():
    assert grade_qualification(INTEL_Q, None) == {"finished": False, "pass": False}
