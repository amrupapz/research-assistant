"""D7 M6: alerts from the run logs. evaluate() is pure; main() reads runs/ and exits 1 on any ALERT."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from test_agent import _settings

from research_assistant import alerts
from research_assistant.alerts import Thresholds, evaluate

T = Thresholds(min_requests=10)


def api(status=200, ms=50, path="/runs/{run_id}"):
    return {"kind": "api", "status": status, "latency_ms": ms, "path": path}


def model(status="completed", ms=2_000):
    return {"model": "gpt-6-luna", "status": status, "latency_ms": ms}


def names(found):
    return [a.split()[1] for a in found[0]]


def test_a_healthy_hour_is_ok():
    assert evaluate([api()] * 50 + [model()] * 50, T)[0] == []


def test_5xx_above_the_threshold_alerts_with_its_evidence():
    found = evaluate([api()] * 18 + [api(status=503)] * 2, T)  # 10% > 5%
    assert names(found) == ["api_error_rate"]
    assert "2/20" in found[0][0] and "10.0% > 5%" in found[0][0]


def test_4xx_are_not_errors_but_429s_have_their_own_alert():
    assert evaluate([api(status=404)] * 20 + [api(status=401)] * 20, T)[0] == []  # clients' mistakes
    assert names(evaluate([api()] * 10 + [api(status=429)] * 10, T)) == ["rate_limited"]


def test_too_few_requests_never_alert_on_a_rate():
    found = evaluate([api(status=500)] * 3 + [model(status="error")] * 3, T)
    assert found[0] == [] and any("need >= 10 samples" in s for s in found[1])


def test_p95_is_per_route():
    slow_ask = [api(path="/ask", ms=12_000)] * 20
    fast_polls = [api(path="/runs/{run_id}", ms=20)] * 200  # would hide the slow route in a global p95
    found = evaluate(slow_ask + fast_polls, T)
    assert names(found) == ["api_p95"] and "/ask 12.0s > 10.0s" in found[0][0]


def test_model_error_rate_alerts_only_above_the_limit():
    calls = [model()] * 18 + [model(status="incomplete"), model(status="error")]
    assert evaluate(calls, T)[0] == []  # 2/20 = 10%: AT the limit, not over
    assert names(evaluate([*calls, model(status="ungrounded")], T)) == ["model_error_rate"]  # 3/21


def test_slow_model_calls():
    assert evaluate([model()] * 19 + [model(ms=40_000)], T)[0] == []  # 1 slow in 20: p95 is the 19th = fast
    assert names(evaluate([model()] * 18 + [model(ms=40_000)] * 2, T)) == ["model_p95"]  # 19th is slow


def test_a_single_crash_or_failed_job_alerts_whatever_the_volume():
    recs = [api()] * 100 + [{"kind": "api_error", "request_id": "r-9"}, {"kind": "api_job", "status": "error", "run_id": "u-1"}]
    found = evaluate(recs, T)
    assert names(found) == ["unhandled_errors", "failed_jobs"]
    assert "r-9" in found[0][0] and "u-1" in found[0][1]


def test_tool_call_lines_are_not_model_calls():
    assert evaluate([{"kind": "tool_call", "status": "denied", "model": "x"}] * 30, T)[0] == []


# --- main(): files, window, exit code, Slack ----------------------------------------------------


@pytest.fixture
def runs(tmp_path, monkeypatch):
    s = _settings(tmp_path / "runs")
    s.runs_dir.mkdir()
    monkeypatch.setattr("research_assistant.config.get_settings", lambda: s)
    now = datetime.now(UTC)

    def write(records, minutes_ago=0):
        ts = (now - timedelta(minutes=minutes_ago)).isoformat()
        with (s.runs_dir / f"{now:%Y-%m-%d}.jsonl").open("a") as f:
            for r in records:
                f.write(json.dumps({"ts": ts, **r}) + "\n")

    write.settings = s
    return write


def test_main_exits_1_and_prints_alert_lines(runs, capsys):
    runs([api(status=500)] * 30)
    with pytest.raises(SystemExit) as e:
        alerts.main([])
    out = capsys.readouterr().out
    assert e.value.code == 1 and out.startswith("ALERT api_error_rate 100.0% > 5%")


def test_main_only_looks_at_the_window(runs, capsys):
    runs([api(status=500)] * 30, minutes_ago=120)  # an old incident
    runs([api()] * 30)
    with pytest.raises(SystemExit) as e:
        alerts.main(["--minutes", "60"])
    assert e.value.code == 0 and capsys.readouterr().out.startswith("OK")


def test_a_truncated_last_line_is_skipped(runs, capsys):
    runs([api()] * 30)
    with (runs.settings.runs_dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl").open("a") as f:
        f.write('{"ts": "2026-11-13T08:00:00+00:00", "kind": "ap')  # crash mid-write
    with pytest.raises(SystemExit) as e:
        alerts.main([])
    assert e.value.code == 0


def test_slack_gets_the_alert_lines_when_configured(runs, monkeypatch):
    posted = []
    monkeypatch.setattr(alerts, "post_slack", lambda url, text: posted.append((url, text)))
    runs([api(status=500)] * 30)
    runs.settings.slack_webhook_url = SecretStr("https://hooks.slack.example/T/B/x")
    with pytest.raises(SystemExit):
        alerts.main([])
    assert posted == [("https://hooks.slack.example/T/B/x", "ALERT api_error_rate 100.0% > 5% (30/30 API responses were 5xx)")]


def test_slack_down_does_not_hide_the_alert(runs, monkeypatch, capsys):
    def down(url, text):
        raise OSError("connection refused")

    monkeypatch.setattr(alerts, "post_slack", down)
    runs([api(status=500)] * 30)
    runs.settings.slack_webhook_url = SecretStr("https://hooks.slack.example/T/B/x")
    with pytest.raises(SystemExit) as e:
        alerts.main([])
    cap = capsys.readouterr()
    assert e.value.code == 1 and "ALERT api_error_rate" in cap.out and "Slack post failed" in cap.err


def test_an_empty_webhook_setting_means_off(runs, monkeypatch):
    monkeypatch.setattr(alerts, "post_slack", lambda url, text: pytest.fail("posted with an empty URL"))
    runs([api(status=500)] * 30)
    runs.settings.slack_webhook_url = SecretStr("")  # SLACK_WEBHOOK_URL= in .env
    with pytest.raises(SystemExit):
        alerts.main([])
