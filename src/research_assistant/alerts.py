"""Health check from the run logs (D7 M6). What each ALERT means and what to do: RUNBOOK.md.

    uv run python -m research_assistant.alerts                      # last 60 minutes, default thresholds
    uv run python -m research_assistant.alerts --minutes 15 --api-p95-ms 5000
    docker compose exec app python -m research_assistant.alerts     # the container's runs/ volume

Prints one `ALERT <name> ...` line per problem, then a summary. Exit 1 if anything alerted (so cron,
CI or a shell `&&` can react). SLACK_WEBHOOK_URL set → the ALERT lines are also posted there.

Reads runs/*.jsonl, the same lines every request and model call already writes:
  kind=api        one per HTTP request (status, latency_ms, path = the ROUTE, so p95 groups by endpoint)
  kind=api_error  an unhandled exception (the client got a 500)
  kind=api_job    a background flow job that raised
  has "model"     one per model / embedding call (status completed | error | incomplete | ...)
Rates need at least --min-requests samples: 1 error out of 2 requests isn't an incident.
"""

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from research_assistant.perf import percentile


@dataclass(frozen=True)
class Thresholds:
    api_error_rate: float = 0.05  # share of API responses that are 5xx
    api_p95_ms: float = 10_000  # per route; /ask (one model call) is ~5 s at p95
    rate_limited_share: float = 0.20  # share of API responses that are 429
    model_error_rate: float = 0.10  # share of model calls that didn't complete
    model_p95_ms: float = 20_000  # single model call
    min_requests: int = 20  # below this, rates are noise: report, don't alert


# --- pure -----------------------------------------------------------------------------------


DEFAULT = Thresholds()


def evaluate(records: list[dict], t: Thresholds = DEFAULT) -> tuple[list[str], list[str]]:
    """(alerts, summary). Each alert is one line: `ALERT <name> <value> > <threshold> (<evidence>)`."""
    alerts: list[str] = []
    api = [r for r in records if r.get("kind") == "api"]
    model = [r for r in records if "model" in r and r.get("kind") != "tool_call"]

    def rate(name: str, bad: int, total: int, limit: float, what: str) -> None:
        if total >= t.min_requests and bad / total > limit:
            alerts.append(f"ALERT {name} {bad / total:.1%} > {limit:.0%} ({bad}/{total} {what})")

    rate("api_error_rate", sum(r.get("status", 0) >= 500 for r in api), len(api), t.api_error_rate, "API responses were 5xx")
    rate("rate_limited", sum(r.get("status") == 429 for r in api), len(api), t.rate_limited_share, "API responses were 429")
    rate("model_error_rate", sum(r.get("status") != "completed" for r in model), len(model), t.model_error_rate, "model calls did not complete")

    for path in sorted({r.get("path", "?") for r in api}):
        lat = [r["latency_ms"] for r in api if r.get("path") == path and r.get("latency_ms") is not None]
        p95 = percentile(lat, 95)
        if len(lat) >= t.min_requests and p95 > t.api_p95_ms:
            alerts.append(f"ALERT api_p95 {path} {p95 / 1000:.1f}s > {t.api_p95_ms / 1000:.1f}s ({len(lat)} requests)")
    model_p95 = percentile([r["latency_ms"] for r in model if r.get("latency_ms") is not None], 95)
    if len(model) >= t.min_requests and model_p95 > t.model_p95_ms:
        alerts.append(f"ALERT model_p95 {model_p95 / 1000:.1f}s > {t.model_p95_ms / 1000:.1f}s ({len(model)} calls)")

    # Counts, not rates: a single one is worth a look, whatever the volume.
    if crashes := [r for r in records if r.get("kind") == "api_error"]:
        alerts.append(f"ALERT unhandled_errors {len(crashes)} (last request_id {crashes[-1].get('request_id')})")
    if jobs := [r for r in records if r.get("kind") == "api_job" and r.get("status") == "error"]:
        alerts.append(f"ALERT failed_jobs {len(jobs)} background run(s) raised (last run_id {jobs[-1].get('run_id')})")

    summary = [
        f"api: {len(api)} requests, {sum(r.get('status', 0) >= 500 for r in api)} 5xx, {sum(r.get('status') == 429 for r in api)} 429",
        (
            f"model: {len(model)} calls, {sum(r.get('status') != 'completed' for r in model)} not completed, "
            f"p95 {'-' if model_p95 is None else f'{model_p95 / 1000:.1f}s'}"
        ),
    ]
    if len(api) < t.min_requests or len(model) < t.min_requests:
        summary.append(f"(rates need >= {t.min_requests} samples; fewer = no rate alert)")
    return alerts, summary


def within(records: list[dict], since: datetime) -> list[dict]:
    return [r for r in records if r.get("ts") and datetime.fromisoformat(r["ts"]) >= since]


# --- I/O ------------------------------------------------------------------------------------


def load(runs_dir: Path, since: datetime) -> list[dict]:
    """Records since `since`, from the day files that can contain them (UTC dates in the file name)."""
    out: list[dict] = []
    for f in sorted(runs_dir.glob("20*.jsonl")):
        if f.stem >= f"{since:%Y-%m-%d}":
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash mid-write
    return within(out, since)


def post_slack(url: str, text: str) -> None:
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), headers={"content-type": "application/json"})
    urllib.request.urlopen(req, timeout=10).close()


def main(argv: list[str] | None = None) -> None:
    from research_assistant.config import get_settings

    d = DEFAULT
    p = argparse.ArgumentParser(prog="alerts", description="Flag error rates and p95 latency from the run logs.", allow_abbrev=False)
    p.add_argument("--minutes", type=int, default=60, help="look back this far (default 60)")
    p.add_argument("--api-error-rate", type=float, default=d.api_error_rate)
    p.add_argument("--api-p95-ms", type=float, default=d.api_p95_ms)
    p.add_argument("--rate-limited-share", type=float, default=d.rate_limited_share)
    p.add_argument("--model-error-rate", type=float, default=d.model_error_rate)
    p.add_argument("--model-p95-ms", type=float, default=d.model_p95_ms)
    p.add_argument("--min-requests", type=int, default=d.min_requests)
    a = p.parse_args(argv)
    t = Thresholds(a.api_error_rate, a.api_p95_ms, a.rate_limited_share, a.model_error_rate, a.model_p95_ms, a.min_requests)

    settings = get_settings()
    since = datetime.now(UTC) - timedelta(minutes=a.minutes)
    alerts, summary = evaluate(load(settings.runs_dir, since), t)
    for line in alerts:
        print(line)
    print(f"{'ALERTS: ' + str(len(alerts)) if alerts else 'OK'}  (last {a.minutes} min of {settings.runs_dir}/)")
    for line in summary:
        print("  " + line)
    if alerts and settings.slack_webhook_url:
        try:
            post_slack(settings.slack_webhook_url.get_secret_value(), "\n".join(alerts))
        except OSError as e:  # the alert is already on stdout; a Slack outage mustn't hide it
            print(f"(Slack post failed: {type(e).__name__})", file=sys.stderr)
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
