"""Cost + latency from trace trees (D6 M4). Pure: plain dicts in (tracing.fetch_observations), numbers out.

Where a run's time goes, by SELF time (an observation's duration minus its direct children's), so
nothing is counted twice:

    model      generation + embedding observations (the app's own model calls)
    retrieval  retriever spans minus the embedding inside them = DB / my search code
    tool       tool spans minus everything nested in them (model, retrieval, approval) = tool code + DB
    human      human-approval spans: reported, SUBTRACTED from app time
    grader     judge_claim (M3): an eval's cost, not the product's: subtracted from app time and $
    our_code   the rest of the roots' time: agent loop, flow nodes, run_steps writes, checkpoints

A trace can have several roots (flow, then flow-approve the next day): only time INSIDE roots counts,
so the wait between them never shows up.
"""

import math

MODEL_TYPES = {"generation", "embedding"}
GRADER_NAMES = {"judge_claim"}
HUMAN_NAMES = {"human-approval"}
BUCKETS = ("model", "retrieval", "tool", "our_code")


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least q% of values <= it. Always an
    observed value. With few values p95 is simply the max (6 cases: p95 = the slowest)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q / 100 * len(ordered))) - 1]


def _ms(o: dict) -> float:
    return (o["end"] - o["start"]) * 1000


def decompose(observations: list[dict]) -> dict:
    """One trace -> {wall_ms, app_ms, model_ms, retrieval_ms, tool_ms, our_code_ms, human_ms,
    grader_ms, cost_usd, grader_cost_usd, model_calls, incomplete, steps}."""
    by_id = {o["id"]: o for o in observations}
    incomplete = any(o["end"] is None for o in observations) or not observations
    done = [o for o in observations if o["end"] is not None]
    children: dict[str, list[dict]] = {}
    for o in done:
        if o["parent_id"] in by_id:
            children.setdefault(o["parent_id"], []).append(o)

    def self_ms(o: dict) -> float:
        return _ms(o) - sum(_ms(c) for c in children.get(o["id"], []))

    roots = [o for o in done if o["parent_id"] not in by_id]
    grader = [o for o in done if o["name"] in GRADER_NAMES]
    model = [o for o in done if o["type"] in MODEL_TYPES and o["name"] not in GRADER_NAMES]
    out = {
        "wall_ms": sum(_ms(r) for r in roots),
        "model_ms": sum(_ms(o) for o in model),
        "retrieval_ms": sum(self_ms(o) for o in done if o["type"] == "retriever"),
        "tool_ms": sum(self_ms(o) for o in done if o["type"] == "tool"),
        "human_ms": sum(_ms(o) for o in done if o["name"] in HUMAN_NAMES),
        "grader_ms": sum(_ms(o) for o in grader),
        "cost_usd": sum(o["cost"] for o in model),
        "tokens": sum(o.get("tokens", 0) for o in model),  # what the gate watches: $ swings with cache timing
        "grader_cost_usd": sum(o["cost"] for o in grader),
        "model_calls": len(model) + len(grader),  # everything that wrote a run-log line (for the cross-check)
        "incomplete": incomplete,
        # per step: FULL duration (search_company = its whole 14s), for p50/p95 per step
        "steps": [(o["name"], o["type"], _ms(o)) for o in done if o["type"] not in ("chain", "agent", "span")],
    }
    out["app_ms"] = out["wall_ms"] - out["human_ms"] - out["grader_ms"]
    out["our_code_ms"] = out["app_ms"] - out["model_ms"] - out["retrieval_ms"] - out["tool_ms"]
    return out


def stage_summary(rows: list[dict]) -> dict:
    """rows = decompose() results for one stage's cases (complete ones only) -> p50/p95 + time split."""
    app = [r["app_ms"] / 1000 for r in rows]
    cost = [r["cost_usd"] for r in rows]
    total_app = sum(r["app_ms"] for r in rows) or 1.0
    return {
        "cases": len(rows),
        "seconds_p50": percentile(app, 50),
        "seconds_p95": percentile(app, 95),
        "usd_p50": percentile(cost, 50),
        "usd_p95": percentile(cost, 95),
        "usd_total": sum(cost),
        "tokens_p50": percentile([r["tokens"] for r in rows], 50),
        "grader_usd_total": sum(r["grader_cost_usd"] for r in rows),
        "human_seconds_total": sum(r["human_ms"] for r in rows) / 1000,
        "model_calls_avg": sum(r["model_calls"] for r in rows) / len(rows) if rows else 0,
        "split": {b: sum(r[f"{b}_ms"] for r in rows) / total_app for b in BUCKETS},
    }


def step_summary(rows: list[dict]) -> list[dict]:
    """p50 / p95 / max per step name across all cases, slowest p95 first."""
    by_name: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        for name, typ, ms in r["steps"]:
            by_name.setdefault((name, typ), []).append(ms / 1000)
    out = [
        {"step": n, "type": t, "count": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v)}
        for (n, t), v in by_name.items()
    ]
    return sorted(out, key=lambda s: -s["p95"])
