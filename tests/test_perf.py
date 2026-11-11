"""D6 M4: cost + latency from trace trees. Hand-built trees (seconds), no network."""

import importlib.util
from pathlib import Path

import pytest

from research_assistant import tracing
from research_assistant.perf import decompose, percentile, stage_summary, step_summary

_spec = importlib.util.spec_from_file_location("eval_report", Path(__file__).parents[1] / "evals" / "report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def o(id, parent, type, name, start, end, cost=0.0):
    return {"id": id, "parent_id": parent, "type": type, "name": name, "start": start, "end": end, "cost": cost}


# agent run, 10s: respond, search_company (retrieval + ask inside), send_email (a human says yes), respond
AGENT = [
    o("r", None, "agent", "agent", 0, 10),
    o("g1", "r", "generation", "respond", 0, 2, cost=0.001),
    o("t1", "r", "tool", "search_company", 2, 8),
    o("ret", "t1", "retriever", "hybrid_search", 2, 3),
    o("e", "ret", "embedding", "embed", 2, 2.5, cost=0.0001),
    o("a", "t1", "generation", "ask", 3, 7, cost=0.002),
    o("t2", "r", "tool", "send_email", 8, 9.5),
    o("h", "t2", "span", "human-approval", 8, 9),
    o("g2", "r", "generation", "respond", 9.5, 9.9, cost=0.0005),
]


def test_agent_time_is_split_by_self_time_and_human_time_is_subtracted():
    d = decompose(AGENT)
    assert d["wall_ms"] == pytest.approx(10_000)
    assert d["human_ms"] == pytest.approx(1_000)
    assert d["app_ms"] == pytest.approx(9_000)  # the human's second is not the app's
    assert d["model_ms"] == pytest.approx(2_000 + 500 + 4_000 + 400)
    assert d["retrieval_ms"] == pytest.approx(1_000 - 500)  # retriever minus its embed = DB/search code
    assert d["tool_ms"] == pytest.approx((6_000 - 1_000 - 4_000) + (1_500 - 1_000))  # self time only
    assert d["our_code_ms"] == pytest.approx(9_000 - 6_900 - 500 - 1_500)
    assert d["cost_usd"] == pytest.approx(0.0036)
    assert (d["model_calls"], d["incomplete"]) == (4, False)


def test_the_buckets_add_up_to_app_time():
    d = decompose(AGENT)
    assert d["model_ms"] + d["retrieval_ms"] + d["tool_ms"] + d["our_code_ms"] == pytest.approx(d["app_ms"])


def test_the_wait_between_flow_and_flow_approve_is_not_counted():
    d = decompose([
        o("f", None, "chain", "flow", 0, 5), o("w", "f", "generation", "write_email", 1, 4, cost=0.001),
        o("fa", None, "chain", "flow-approve", 86_400, 86_401),  # approved the next day
    ])  # fmt: skip
    assert d["wall_ms"] == pytest.approx(6_000)


def test_the_judge_is_eval_cost_not_app_time_or_app_cost():
    d = decompose([
        o("root", None, "chain", "eval/answers/q06", 0, 10),
        o("a", "root", "generation", "ask", 1, 4, cost=0.002),
        o("j", "root", "generation", "judge_claim", 5, 9, cost=0.01),
    ])  # fmt: skip
    assert (d["grader_ms"], d["app_ms"], d["model_ms"]) == pytest.approx((4_000, 6_000, 3_000))
    assert (d["cost_usd"], d["grader_cost_usd"]) == pytest.approx((0.002, 0.01))
    assert d["model_calls"] == 2  # both wrote a run-log line


def test_an_unfinished_trace_is_incomplete():
    assert decompose([o("r", None, "agent", "agent", 0, None)])["incomplete"]
    assert decompose([])["incomplete"]


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [(list(range(1, 11)), 50, 5), (list(range(1, 11)), 95, 10), ([3.0], 95, 3.0), ([1, 2, 3, 4, 5, 6], 95, 6), ([], 50, None)],
)
def test_nearest_rank_percentile(values, q, expected):
    assert percentile(values, q) == expected


def test_stage_and_step_summaries():
    rows = [decompose(AGENT), decompose(AGENT)]
    st = stage_summary(rows)
    assert st["cases"] == 2 and st["seconds_p50"] == pytest.approx(9.0)
    assert sum(st["split"].values()) == pytest.approx(1.0)
    steps = {s["step"]: s for s in step_summary(rows)}
    assert steps["search_company"]["p95"] == pytest.approx(6.0)  # FULL duration per step
    assert "agent" not in steps  # roots/chains aren't steps


def test_langfuse_cost_without_a_total_key_is_summed():
    """Found live: cached-token calls came back {input, output, input_cached_tokens} with no total."""
    assert tracing._cost({"input": 4.67e-05, "output": 9.55e-05, "input_cached_tokens": 1.447e-05}) == pytest.approx(1.5667e-4)
    assert tracing._cost({"input": 1.0, "output": 2.0, "total": 3.0}) == 3.0
    assert tracing._cost({}) == 0.0


# --- evals/report.py (pure parts) ---------------------------------------------------------------


def test_select_batch_prefers_the_latest_full_run():
    recs = [
        {"batch": "b1", "stage": "answers", "case": "q1"}, {"batch": "b1", "stage": "all"},
        {"batch": "b2", "stage": "agent", "case": "a1"},  # a later single-stage run
    ]  # fmt: skip
    assert report.select_batch(recs) == ("b1", [recs[0]])
    assert report.select_batch(recs, "b2") == ("b2", [recs[2]])


def test_log_call_counts_skip_tool_lines():
    lines = [{"trace_id": "t", "kind": "agent_turn"}, {"trace_id": "t", "kind": "tool_call"}, {"trace_id": "t"}, {}]
    assert report.log_call_counts(lines) == {"t": 2}


def test_a_trace_missing_calls_is_flagged_not_averaged():
    full = {"stage": "agent", "case": "a1", "decomposition": decompose(AGENT), "expected_calls": 4}
    lagging = {"stage": "agent", "case": "a2", "decomposition": decompose(AGENT), "expected_calls": 6}
    rep = report.build_report([full, lagging])
    assert rep["stages"]["agent"]["cases"] == 1
    assert rep["incomplete"] == [{"stage": "agent", "case": "a2", "trace_calls": 4, "log_calls": 6}]
    text = report.format_report(rep)
    assert "WHERE THE TIME GOES" in text and "INCOMPLETE" in text


def test_format_trace_shows_excluded_human_time():
    assert "human approval (excluded)       1.00s" in report.format_trace("t", decompose(AGENT))
