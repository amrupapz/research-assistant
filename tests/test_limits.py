"""D7 M3: the pieces behind the 429s. The rate limiter (pure), the token meter in llm.py, the
Runner writing usage rows, and the "today" queries. The HTTP behaviour is in test_api.py."""

from contextlib import contextmanager
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from test_agent import _settings

from research_assistant import api, llm, state
from research_assistant.db import set_tenant
from research_assistant.tools import ToolContext

# --- RateLimiter ----------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_the_limit_allows_n_then_says_how_long_to_wait():
    clock = Clock()
    rl = api.RateLimiter(3, clock)
    assert [rl.check(1) for _ in range(3)] == [None, None, None]
    clock.t += 10
    assert rl.check(1) == pytest.approx(50)  # the first hit leaves the window 60 s after it came


def test_the_window_slides():
    clock = Clock()
    rl = api.RateLimiter(2, clock)
    rl.check(1)
    clock.t += 30
    rl.check(1)
    clock.t += 30.5  # the first hit is now older than 60 s
    assert rl.check(1) is None
    assert rl.check(1) is not None  # but the second one is still inside


def test_keys_are_limited_separately():
    rl = api.RateLimiter(1, Clock())
    assert rl.check(1) is None and rl.check(1) is not None
    assert rl.check(2) is None


def test_a_refused_request_does_not_count():
    """Otherwise a client that keeps retrying would never get back under the limit."""
    clock = Clock()
    rl = api.RateLimiter(1, clock)
    rl.check(1)
    for _ in range(5):
        rl.check(1)
    clock.t += 60.1
    assert rl.check(1) is None


# --- token meter ----------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path):
    return _settings(tmp_path / "runs")


def test_the_meter_sums_input_and_output_tokens(settings):
    with llm.token_meter() as used:
        llm._log({"input_tokens": 100, "output_tokens": 20}, settings)
        llm._log({"input_tokens": 5, "output_tokens": None}, settings)  # an embedding call
        llm._log({"status": "error"}, settings)  # failed before any usage came back
    assert used[0] == 125


def test_no_meter_no_counting_and_meters_do_not_leak(settings):
    llm._log({"input_tokens": 100, "output_tokens": 20}, settings)  # no active meter: fine
    with llm.token_meter() as first:
        pass
    with llm.token_meter() as second:
        llm._log({"input_tokens": 1, "output_tokens": 1}, settings)
    assert (first[0], second[0]) == (0, 2)


def test_the_meter_sees_calls_made_inside_langgraph_nodes(settings):
    """LangGraph runs nodes in worker threads with a COPY of the context. The copy still points at
    the same box, so the tokens land in the meter the Runner opened."""

    class S(TypedDict):
        n: int

    def node(s: S) -> S:
        llm._log({"input_tokens": 7, "output_tokens": 3}, settings)
        return {"n": s["n"] + 1}

    g = StateGraph(S)
    g.add_node("a", node)
    g.add_node("b", node)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    with llm.token_meter() as used:
        g.compile().invoke({"n": 0})
    assert used[0] == 20


# --- Runner writes one usage row per job ------------------------------------------------------


@pytest.fixture
def ctx(state_db, tmp_path):
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


def _runner(settings, ctx, monkeypatch) -> api.Runner:
    runner = api.Runner(settings)

    @contextmanager
    def on_the_test_connection(run_id, tenant_id):
        yield ctx, None

    monkeypatch.setattr(runner, "_ctx", on_the_test_connection)
    return runner


def _usage(conn) -> list[tuple]:
    return conn.execute("SELECT tenant_id, endpoint, tokens, run_id FROM usage ORDER BY usage_id").fetchall()


def test_a_job_writes_its_tokens_once(settings, ctx, monkeypatch):
    run_id = state.start_run(ctx.conn, "flow L1", "m", "flow-v2")

    def job(c, cp):
        llm._log({"input_tokens": 100, "output_tokens": 5}, settings)
        llm._log({"input_tokens": 50, "output_tokens": 5}, settings)

    _runner(settings, ctx, monkeypatch)._job(run_id, "default", job)
    assert _usage(ctx.conn) == [("default", "runs", 160, run_id)]


def test_a_failed_job_still_records_what_it_spent(settings, ctx, monkeypatch):
    run_id = state.start_run(ctx.conn, "flow L1", "m", "flow-v2")

    def job(c, cp):
        llm._log({"input_tokens": 100, "output_tokens": 5}, settings)
        raise RuntimeError("model went away")

    _runner(settings, ctx, monkeypatch)._job(run_id, "default", job)  # logs, doesn't raise
    assert [r[2] for r in _usage(ctx.conn)] == [105]


def test_a_job_without_model_calls_writes_no_row(settings, ctx, monkeypatch):
    run_id = state.start_run(ctx.conn, "flow L1", "m", "flow-v2")
    _runner(settings, ctx, monkeypatch)._job(run_id, "default", lambda c, cp: None)  # e.g. a decline
    assert _usage(ctx.conn) == []


# --- "today", per tenant -----------------------------------------------------------------------


def test_today_means_since_utc_midnight_and_only_my_tenant(ctx):
    conn = ctx.conn
    conn.execute("INSERT INTO tenants (tenant_id) VALUES ('acme') ON CONFLICT DO NOTHING")
    state.record_usage(conn, 300, "ask")
    state.record_usage(conn, 5_000, "ask")
    conn.execute("UPDATE usage SET created_at = now() - interval '1 day' WHERE tokens = 5000")  # yesterday
    set_tenant(conn, "acme")
    state.record_usage(conn, 7, "ask")
    assert state.tokens_today(conn) == 7
    set_tenant(conn, "default")
    assert state.tokens_today(conn) == 300
    state.start_run(conn, "flow L1", "m", "flow-v2")
    assert state.runs_today(conn) == 1


def test_new_tenants_get_the_default_plan(ctx):
    ctx.conn.execute("INSERT INTO tenants (tenant_id) VALUES ('acme') ON CONFLICT DO NOTHING")
    set_tenant(ctx.conn, "acme")
    assert state.tenant_limits(ctx.conn) == (50, 1_000_000)
