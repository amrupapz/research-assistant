"""Cost + latency report from traces (D6 M4).

  uv run python evals/report.py                 # the latest `evals/run.py --all` batch
  uv run python evals/report.py --batch 20261111T101500
  uv run python evals/report.py --trace d7f0adaa9293e95e81883a063455546d   # one real CLI run

Reads the trace ids each eval case recorded in runs/evals.jsonl, fetches the trees from Langfuse
(cached in runs/traces/, so a re-run is offline and gives the same numbers), splits time with
perf.decompose, and writes runs/report.json (M5's baseline reads it).

Cross-check: a case whose trace has a different number of model calls than the run log says
for that trace_id is INCOMPLETE (usually Langfuse ingestion lag): listed, not averaged in.
"""

import argparse
import json
from pathlib import Path

from research_assistant import perf, tracing

ROOT = Path(__file__).parent.parent
EVALS = ROOT / "runs" / "evals.jsonl"
CACHE = ROOT / "runs" / "traces"
OUT = ROOT / "runs" / "report.json"
STAGES = ("answers", "qualify", "agent", "flow")


# --- pure ---------------------------------------------------------------------------------------


def select_batch(records: list[dict], batch: str | None = None) -> tuple[str | None, list[dict]]:
    """The records of one batch: the given one, else the latest that ran `--all`, else the latest."""
    if batch is None:
        full = [r["batch"] for r in records if r.get("stage") == "all" and r.get("batch")]
        batches = [r["batch"] for r in records if r.get("batch")]
        batch = (full or batches or [None])[-1]
    return batch, [r for r in records if r.get("batch") == batch and r.get("stage") in STAGES]


def log_call_counts(log_lines: list[dict]) -> dict[str, int]:
    """trace_id -> model calls the RUN LOG saw (every non-tool line with a trace_id)."""
    counts: dict[str, int] = {}
    for r in log_lines:
        if r.get("trace_id") and r.get("kind") != "tool_call":
            counts[r["trace_id"]] = counts.get(r["trace_id"], 0) + 1
    return counts


def build_report(cases: list[dict]) -> dict:
    """cases = [{stage, case, decomposition, expected_calls}] -> stages, steps, top cases, incomplete."""
    ok, incomplete = [], []
    for c in cases:
        d = c["decomposition"]
        if d["incomplete"] or (c["expected_calls"] is not None and d["model_calls"] != c["expected_calls"]):
            incomplete.append({"stage": c["stage"], "case": c["case"], "trace_calls": d["model_calls"],
                               "log_calls": c["expected_calls"]})  # fmt: skip
        else:
            ok.append(c)
    by_stage = {s: [c["decomposition"] for c in ok if c["stage"] == s] for s in STAGES}

    def top(key: str) -> list[dict]:
        ranked = sorted(ok, key=lambda c: -c["decomposition"][key])[:5]
        return [{"stage": c["stage"], "case": c["case"], key: c["decomposition"][key]} for c in ranked]

    return {
        "stages": {s: perf.stage_summary(rows) for s, rows in by_stage.items() if rows},
        "steps": perf.step_summary([c["decomposition"] for c in ok]),
        "slowest": top("app_ms"),
        "priciest": top("cost_usd"),
        "incomplete": incomplete,
        "usd_total": sum(c["decomposition"]["cost_usd"] for c in ok),
        "grader_usd_total": sum(c["decomposition"]["grader_cost_usd"] for c in ok),
    }


def _s(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def format_report(rep: dict) -> str:
    lines = ["", "PER STAGE (app time = wall - human approval - grader; $ excludes the grader)"]
    lines.append(f"{'stage':<9}{'cases':>6}{'s p50':>8}{'s p95':>8}{'$ p50':>10}{'$ p95':>10}{'$ total':>10}{'calls':>7}")
    for s, st in rep["stages"].items():
        lines.append(
            f"{s:<9}{st['cases']:>6}{_s(st['seconds_p50']):>8}{_s(st['seconds_p95']):>8}"
            f"{st['usd_p50']:>10.5f}{st['usd_p95']:>10.5f}{st['usd_total']:>10.4f}{st['model_calls_avg']:>7.1f}"
        )
    lines += ["", "WHERE THE TIME GOES (share of app time)"]
    lines.append(f"{'stage':<9}" + "".join(f"{b:>11}" for b in perf.BUCKETS) + f"{'human s':>9}")
    for s, st in rep["stages"].items():
        lines.append(f"{s:<9}" + "".join(f"{st['split'][b]:>11.0%}" for b in perf.BUCKETS) + f"{st['human_seconds_total']:>9.1f}")
    lines += ["", "PER STEP (full duration, seconds; with few calls p95 = max)"]
    lines.append(f"{'step':<17}{'type':<11}{'n':>4}{'p50':>7}{'p95':>7}{'max':>7}")
    for st in rep["steps"]:
        lines.append(f"{st['step']:<17}{st['type']:<11}{st['count']:>4}{st['p50']:>7.2f}{st['p95']:>7.2f}{st['max']:>7.2f}")
    lines += ["", "SLOWEST CASES"] + [f"  {c['stage']:<8}{c['case']:<6}{c['app_ms'] / 1000:>7.1f}s" for c in rep["slowest"]]
    lines += ["", "MOST EXPENSIVE CASES"] + [f"  {c['stage']:<8}{c['case']:<6}${c['cost_usd']:.5f}" for c in rep["priciest"]]
    lines += ["", f"TOTAL app ${rep['usd_total']:.4f}  |  grader (judge) ${rep['grader_usd_total']:.4f}"]
    if rep["incomplete"]:
        lines += ["", "INCOMPLETE (not in the numbers: trace calls != run-log calls; re-run the report later)"]
        lines += [f"  {i['stage']:<8}{i['case']:<6} trace={i['trace_calls']} log={i['log_calls']}" for i in rep["incomplete"]]
    return "\n".join(lines)


def format_trace(trace_id: str, d: dict) -> str:
    rows = [("app (wall - human - grader)", d["app_ms"]), ("  model", d["model_ms"]), ("  retrieval / DB", d["retrieval_ms"]),
            ("  tool code + DB", d["tool_ms"]), ("  our code", d["our_code_ms"]), ("human approval (excluded)", d["human_ms"]),
            ("grader (excluded)", d["grader_ms"])]  # fmt: skip
    app = d["app_ms"] or 1.0
    out = [f"\ntrace {trace_id}  ({d['model_calls']} model calls, ${d['cost_usd']:.5f})"]
    out += [f"  {name:<28}{ms / 1000:>8.2f}s" + (f"{ms / app:>6.0%}" if name.startswith("  ") else "") for name, ms in rows]
    return "\n".join(out)


# --- I/O ----------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observations(trace_id: str) -> list[dict]:
    """From the cache, else Langfuse. Only COMPLETE trees are cached (a lagging one is re-fetched)."""
    cached = CACHE / f"{trace_id}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    obs = tracing.fetch_observations(trace_id)
    if obs and all(o["end"] is not None for o in obs):
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(obs))
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost + latency report from traces (D6 M4).")
    parser.add_argument("--batch", help="an evals/run.py batch id (default: latest --all)")
    parser.add_argument("--trace", nargs="*", help="report single trace ids instead (e.g. a real CLI run)")
    args = parser.parse_args()
    try:
        if args.trace:
            for t in args.trace:
                print(format_trace(t, perf.decompose(observations(t))))
            return
        batch, records = select_batch(_read_jsonl(EVALS) if EVALS.exists() else [], args.batch)
        traced = [r for r in records if r.get("trace_ids")]
        if not traced:
            raise SystemExit(f"batch {batch}: no traced cases (run evals with TRACING_ENABLED=true)")
        logs = [line for f in sorted((ROOT / "runs").glob("20*.jsonl")) for line in _read_jsonl(f)]
        counts = log_call_counts(logs)
        cases = []
        for r in traced:
            obs = [o for t in r["trace_ids"] for o in observations(t)]
            cases.append({
                "stage": r["stage"], "case": r["case"], "decomposition": perf.decompose(obs),
                "expected_calls": sum(counts.get(t, 0) for t in r["trace_ids"]),
            })  # fmt: skip
        rep = build_report(cases)
        print(f"batch {batch}: {len(cases)} traced cases ({len(records) - len(traced)} untraced)")
        print(format_report(rep))
        OUT.write_text(json.dumps({"batch": batch, **rep}, indent=2, default=str))
        print(f"\nsaved {OUT.relative_to(ROOT)}")
    finally:
        tracing.shutdown()


if __name__ == "__main__":
    main()
