"""Regression gate (D6 M5): is this eval run worse than the committed baseline? Pure: no I/O.

A SNAPSHOT (evals/baseline.json, and the "current" run) looks like:

    {
      "batch": "20261111T070332",
      "versions": {"prompt": "v4", "agent": "agent-v5", "flow": "flow-v2", "judge": "judge-v3", "icp": "icp-v1"},
      "categories": {"freshness": 0.7, "prompt_injection": 0.6, ...},      # pass rate, 0.0 .. 1.0
      "stages": {"agent": {"seconds_p95": 20.8, "tokens_p50": 9500, "usd_p50": 0.00065}, ...}  # runs/report.json

    The gate checks tokens_p50, not usd_p50: the dollar cost of the same run swung +83% with prompt-cache
    timing alone (M5). usd_p50 stays in the snapshot for humans.
    }
"""

DEFAULT_TOLERANCE = {
    "pass_rate": 0.10,  # ABSOLUTE drop allowed per category: 0.90 -> 0.80 is ok, -> 0.79 is not
    "p95": 0.25,  # RELATIVE rise allowed in a stage's seconds_p95: 20s -> 25s ok, -> 25.1s not
    "cost": 0.20,  # RELATIVE rise allowed in a stage's tokens_p50 (the cost driver)
}


def regression_gate(baseline: dict, current: dict, tolerance: dict = DEFAULT_TOLERANCE) -> tuple[bool, list[str]]:
    """Compare two snapshots; return (ok, reasons). ok is True exactly when reasons is empty.

    ✍️ Project 6 core function. Written by Claude at my request (I overrode my "hint only" rule).

    Contract:
      Quality, for EVERY category in baseline["categories"]:
        - missing from current["categories"]          -> a reason (a category that vanished = untested)
        - current rate < baseline rate - pass_rate     -> a reason
      Speed and cost, for EVERY stage in baseline["stages"]:
        - missing from current["stages"]                         -> a reason
        - current seconds_p95 > baseline seconds_p95 * (1 + p95) -> a reason
        - current tokens_p50  > baseline tokens_p50  * (1 + cost) -> a reason
        - a baseline value of 0 (or None) has no relative limit: skip that check
      Trace gaps (D7 M5: these used to pass SILENTLY):
        - current["incomplete"] non-empty -> a reason (report.py dropped those cases, so p95/tokens
          would come from fewer cases than the baseline)
        - a metric the baseline has (non-zero) but current lacks (None) -> a reason, not a skip
      Never a reason:
        - anything that got BETTER, or is exactly AT the limit (the comparisons are strict)
        - categories or stages that exist only in current (new tests aren't regressions)
      Each reason is ONE string naming the category or stage, the metric, and both numbers,
      e.g. "freshness: pass rate 0.70 -> 0.50 (allowed down to 0.60)". Tests check that the name
      and the metric appear, not the exact wording.

    Hints:
      - Collect reasons in a list; return (not reasons, reasons).
      - Two small loops: one over categories, one over stages x [("seconds_p95", "p95"), ("tokens_p50", "cost")].
      - .get() for the "missing" cases. Floating-point: compare with the computed limit, don't round.
      - ~20 lines. Run: uv run pytest tests/test_gate.py -v
    """
    reasons: list[str] = []

    for name, base_rate in baseline["categories"].items():
        cur_rate = current["categories"].get(name)
        if cur_rate is None:
            reasons.append(f"{name}: category missing from the current run (was {base_rate:.2f})")
            continue
        floor = base_rate - tolerance["pass_rate"]
        if cur_rate < floor:
            reasons.append(f"{name}: pass rate {base_rate:.2f} -> {cur_rate:.2f} (allowed down to {floor:.2f})")

    for stage, base_metrics in baseline["stages"].items():
        cur_metrics = current["stages"].get(stage)
        if cur_metrics is None:
            reasons.append(f"{stage}: stage missing from the current run")
            continue
        for metric, tol_key in (("seconds_p95", "p95"), ("tokens_p50", "cost")):
            base_value = base_metrics.get(metric)
            if not base_value:
                continue
            limit = base_value * (1 + tolerance[tol_key])
            cur_value = cur_metrics.get(metric)
            if cur_value is None:
                reasons.append(f"{stage}: {metric} missing from the current run (baseline {base_value:.5g})")
            elif cur_value > limit:
                reasons.append(f"{stage}: {metric} {base_value:.5g} -> {cur_value:.5g} (allowed up to {limit:.5g})")

    incomplete = current.get("incomplete") or []
    if incomplete:
        shown = ", ".join(incomplete[:5]) + (" ..." if len(incomplete) > 5 else "")
        reasons.append(
            f"{len(incomplete)} case(s) incomplete in Langfuse ({shown}): speed/cost would be compared on "
            "fewer cases; re-run evals/report.py in a few minutes"
        )

    return not reasons, reasons


def snapshot(all_record: dict, report: dict, versions: dict) -> dict:
    """Build a snapshot from an `evals/run.py --all` record (its per-category rows) and the matching
    runs/report.json (per-stage p95 + $). Raises if they're from different batches."""
    if report.get("batch") != all_record.get("batch"):
        raise ValueError(
            f"report is batch {report.get('batch')}, evals run is {all_record.get('batch')}: run evals/report.py first"
        )
    return {
        "batch": all_record["batch"],
        "versions": versions,
        "incomplete": [f"{i['stage']}/{i['case']}" for i in report.get("incomplete", [])],
        "categories": {row["category"]: row["passed"] / row["total"] for row in all_record["categories"] if row["total"]},
        "stages": {
            name: {"seconds_p95": st["seconds_p95"], "tokens_p50": st.get("tokens_p50"), "usd_p50": st["usd_p50"]}
            for name, st in report["stages"].items()
        },
    }
