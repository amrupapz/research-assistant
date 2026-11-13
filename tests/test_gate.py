"""D6 M5: the regression gate. The regression_gate tests are the spec for YOUR function: they fail
until you write it. snapshot() is mine and should already pass."""

import pytest

from research_assistant.gate import DEFAULT_TOLERANCE, regression_gate, snapshot

BASE = {
    "categories": {"freshness": 0.70, "prompt_injection": 0.60, "straightforward": 1.0},
    "stages": {
        "agent": {"seconds_p95": 20.0, "tokens_p50": 10_000, "usd_p50": 0.0010},
        "answers": {"seconds_p95": 5.0, "tokens_p50": 3_000, "usd_p50": 0.0003},
    },
}


def current(**changes):
    """BASE with nested overrides, e.g. current(cat_freshness=0.5, agent_seconds_p95=30)."""
    cur = {"categories": dict(BASE["categories"]), "stages": {k: dict(v) for k, v in BASE["stages"].items()}}
    for key, value in changes.items():
        kind, rest = key.split("_", 1)
        if kind == "cat":
            cur["categories"][rest] = value
        else:
            stage, metric = rest.split("_", 1) if kind == "stage" else (kind, rest)
            cur["stages"][stage][metric] = value
    return cur


def _reasons_mention(reasons, *words):
    return any(all(w in r for w in words) for r in reasons)


# --- quality ---------------------------------------------------------------------------------------


def test_identical_runs_pass():
    assert regression_gate(BASE, current()) == (True, [])


def test_a_category_dropping_past_the_tolerance_fails():
    ok, reasons = regression_gate(BASE, current(cat_freshness=0.55))
    assert not ok and len(reasons) == 1
    assert _reasons_mention(reasons, "freshness")


def test_a_drop_exactly_at_the_tolerance_passes():
    ok, _ = regression_gate(BASE, current(cat_freshness=0.60))  # 0.70 - 0.10: at the limit, not past it
    assert ok


def test_a_missing_category_fails():
    cur = current()
    del cur["categories"]["prompt_injection"]
    ok, reasons = regression_gate(BASE, cur)
    assert not ok and _reasons_mention(reasons, "prompt_injection")


def test_improvements_and_new_categories_never_fail():
    cur = current(cat_freshness=1.0)
    cur["categories"]["brand_new"] = 0.0  # a new, failing category is not a REGRESSION
    assert regression_gate(BASE, cur) == (True, [])


# --- speed and cost -------------------------------------------------------------------------------


def test_p95_rising_past_the_tolerance_fails():
    ok, reasons = regression_gate(BASE, current(agent_seconds_p95=25.1))  # limit 20 * 1.25 = 25
    assert not ok and _reasons_mention(reasons, "agent", "p95")


def test_p95_at_the_limit_passes():
    assert regression_gate(BASE, current(agent_seconds_p95=25.0))[0]


def test_tokens_rising_past_the_tolerance_fails():
    ok, reasons = regression_gate(BASE, current(answers_tokens_p50=3_601))  # limit 3000 * 1.2 = 3600
    assert not ok and _reasons_mention(reasons, "answers", "tokens")


def test_dollars_alone_never_fail_the_gate():
    """Same tokens, 2x the bill = prompt-cache timing (M5's false alarm), not a regression."""
    assert regression_gate(BASE, current(answers_usd_p50=0.0006)) == (True, [])


def test_a_missing_stage_fails():
    cur = current()
    del cur["stages"]["answers"]
    ok, reasons = regression_gate(BASE, cur)
    assert not ok and _reasons_mention(reasons, "answers")


def test_a_zero_baseline_has_no_relative_limit():
    base = {"categories": {}, "stages": {"flow": {"seconds_p95": 0.0, "tokens_p50": None}}}
    cur = {"categories": {}, "stages": {"flow": {"seconds_p95": 3.0, "tokens_p50": 5_000}}}
    assert regression_gate(base, cur) == (True, [])


def test_every_problem_is_reported_not_just_the_first():
    ok, reasons = regression_gate(BASE, current(cat_freshness=0.1, agent_seconds_p95=99, agent_tokens_p50=99_999))
    assert not ok and len(reasons) == 3


def test_tolerance_is_a_parameter():
    strict = {**DEFAULT_TOLERANCE, "pass_rate": 0.0}
    assert not regression_gate(BASE, current(cat_freshness=0.69), strict)[0]
    assert regression_gate(BASE, current(cat_freshness=0.69))[0]  # default 0.10 allows it


# --- trace gaps (D7 M5): these used to pass silently ---------------------------------------------


def test_incomplete_cases_fail_the_gate():
    ok, reasons = regression_gate(BASE, {**current(), "incomplete": ["agent/A3", "flow/F1"]})
    assert not ok and _reasons_mention(reasons, "2 case(s) incomplete", "agent/A3")


def test_no_incomplete_key_is_fine():
    """baseline.json files from before this change have no "incomplete" key."""
    assert regression_gate(BASE, {k: v for k, v in current().items() if k != "incomplete"}) == (True, [])


def test_a_metric_missing_from_the_current_run_fails():
    ok, reasons = regression_gate(BASE, current(agent_tokens_p50=None))
    assert not ok and _reasons_mention(reasons, "agent", "tokens_p50", "missing")


def test_a_metric_missing_from_the_baseline_is_still_skipped():
    base = {**BASE, "stages": {**BASE["stages"], "agent": {**BASE["stages"]["agent"], "tokens_p50": None}}}
    assert regression_gate(base, current(agent_tokens_p50=None)) == (True, [])


def test_snapshot_keeps_the_incomplete_cases():
    rec = {"batch": "b1", "categories": []}
    rep = {"batch": "b1", "stages": {}, "incomplete": [{"stage": "flow", "case": "F2", "trace_calls": 1, "log_calls": 3}]}
    assert snapshot(rec, rep, {})["incomplete"] == ["flow/F2"]


# --- snapshot (mine) ------------------------------------------------------------------------------


def test_snapshot_from_an_all_record_and_its_report():
    rec = {"batch": "b1", "categories": [{"category": "freshness", "passed": 7, "total": 10}]}
    rep = {"batch": "b1", "stages": {"agent": {"seconds_p95": 20.8, "tokens_p50": 9500, "usd_p50": 0.00065, "cases": 24}}}
    snap = snapshot(rec, rep, {"icp": "icp-v1"})
    assert snap == {
        "batch": "b1",
        "versions": {"icp": "icp-v1"},
        "incomplete": [],
        "categories": {"freshness": 0.7},
        "stages": {"agent": {"seconds_p95": 20.8, "tokens_p50": 9500, "usd_p50": 0.00065}},
    }


def test_snapshot_refuses_a_report_from_another_batch():
    with pytest.raises(ValueError, match="run evals/report.py first"):
        snapshot({"batch": "b2", "categories": []}, {"batch": "b1", "stages": {}}, {})
