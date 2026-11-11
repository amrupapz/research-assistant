"""Regression gate CLI (D6 M5). Day 7's GitHub Actions runs exactly this.

  uv run python evals/run.py --all --repeat 3 && uv run python evals/report.py
  uv run python evals/gate.py --save-baseline      # freeze it -> evals/baseline.json (commit it)
  ... change something, re-run the two commands above ...
  uv run python evals/gate.py                      # exit 0 = ok, exit 1 = regression (reasons printed)
"""

import argparse
import json
import sys
from pathlib import Path

from research_assistant.flow import FLOW_VERSION
from research_assistant.gate import DEFAULT_TOLERANCE, regression_gate, snapshot
from research_assistant.prompts import (
    AGENT_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    PROMPT_VERSION,
)
from research_assistant.qualify import load_icp

ROOT = Path(__file__).parent.parent
EVALS = ROOT / "runs" / "evals.jsonl"
REPORT = ROOT / "runs" / "report.json"
BASELINE = ROOT / "evals" / "baseline.json"


def versions() -> dict:
    return {
        "prompt": PROMPT_VERSION,
        "agent": AGENT_PROMPT_VERSION,
        "flow": FLOW_VERSION,
        "judge": JUDGE_PROMPT_VERSION,
        "icp": load_icp(ROOT / "icp.toml").version,
    }


def latest_snapshot() -> dict:
    """The newest `--all` run + the report built from it."""
    alls = [r for r in map(json.loads, EVALS.read_text().splitlines()) if r.get("stage") == "all"] if EVALS.exists() else []
    if not alls:
        raise SystemExit("no `evals/run.py --all` run found in runs/evals.jsonl")
    if not REPORT.exists():
        raise SystemExit("no runs/report.json: run evals/report.py first")
    return snapshot(alls[-1], json.loads(REPORT.read_text()), versions())


def side_by_side(base: dict, cur: dict) -> str:
    lines = [f"{'':<24}{'baseline':>10}{'current':>10}"]
    for cat in sorted(set(base["categories"]) | set(cur["categories"])):
        b, c = base["categories"].get(cat), cur["categories"].get(cat)
        lines.append(f"{cat:<24}{'-' if b is None else f'{b:.0%}':>10}{'-' if c is None else f'{c:.0%}':>10}")
    for stage in sorted(set(base["stages"]) | set(cur["stages"])):
        for metric in ("seconds_p95", "usd_p50"):
            b, c = base["stages"].get(stage, {}).get(metric), cur["stages"].get(stage, {}).get(metric)
            lines.append(f"{stage + ' ' + metric:<24}{'-' if b is None else f'{b:.5g}':>10}{'-' if c is None else f'{c:.5g}':>10}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the latest eval run against evals/baseline.json.")
    parser.add_argument("--save-baseline", action="store_true", help="freeze the latest run as the baseline")
    args = parser.parse_args()
    cur = latest_snapshot()
    if args.save_baseline:
        BASELINE.write_text(json.dumps(cur, indent=2) + "\n")
        print(f"saved {BASELINE.relative_to(ROOT)} from batch {cur['batch']} ({cur['versions']})")
        return
    if not BASELINE.exists():
        raise SystemExit("no evals/baseline.json: run with --save-baseline first")
    base = json.loads(BASELINE.read_text())
    changed = {k: (base["versions"].get(k), v) for k, v in cur["versions"].items() if base["versions"].get(k) != v}
    print(f"baseline batch {base['batch']}  vs  current batch {cur['batch']}")
    if changed:
        print("versions changed: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items()))
    print(side_by_side(base, cur))
    try:
        ok, reasons = regression_gate(base, cur, DEFAULT_TOLERANCE)
    except NotImplementedError as e:
        raise SystemExit(f"\n{e}") from None
    print(f"\nGATE: {'PASS' if ok else 'FAIL'}")
    for r in reasons:
        print(f"  ✗ {r}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
