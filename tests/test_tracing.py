"""D6 M1: Langfuse tracing. Real SDK + in-memory exporter (`spans` fixture): nothing is sent.

What must hold: one trace per run (trace_id seeded from run_id, stored in runs + run log), model
calls carry tokens + cost, tool calls carry the gate's decision, a paused-then-approved flow is
ONE trace, and with tracing off nothing is built at all.
"""

import json
import uuid

import pytest
from pydantic import SecretStr
from test_agent import FakeModel, _runs, call, final, turn
from test_ask import CHUNKS, COMPLETED, _client, _read_runs
from test_ask import _settings as _ask_settings
from test_flow import STRONG, ctx, fake, saver, settings  # noqa: F401 -- fixtures, used by name

from research_assistant import flow, state, tools, tracing
from research_assistant.agent import run_agent
from research_assistant.config import Settings
from research_assistant.flow import approve_flow, run_flow
from research_assistant.llm import ask
from research_assistant.permissions import authorize

# --- pure helpers ---------------------------------------------------------------------------


def test_mask_replaces_email_addresses_everywhere():
    data = {"to": "dana.kim@example.com", "body": ["Write to a+b@x.co.uk please", 3], "n": None}
    assert tracing.mask(data=data) == {"to": "<email>", "body": ["Write to <email> please", 3], "n": None}


def test_usage_buckets_do_not_overlap():
    """OpenAI's input_tokens INCLUDES cached tokens: sending both as-is would double-count them."""
    assert tracing.usage_details({"input_tokens": 1000, "cached_tokens": 800, "output_tokens": 50}) == {
        "input": 200,
        "input_cached_tokens": 800,
        "output": 50,
    }
    assert tracing.usage_details({"input_tokens": 10, "output_tokens": None}) == {"input": 10, "output": 0}


def test_cost_comes_from_the_price_table_per_million_tokens():
    prices = {"m": {"input": 2.0, "cached_input": 0.5, "output": 8.0}}
    cost = tracing.cost_details("m", {"input": 200, "input_cached_tokens": 800, "output": 50}, prices)
    assert cost == pytest.approx({"input": 0.0004, "input_cached_tokens": 0.0004, "output": 0.0004})
    assert tracing.cost_details("unpriced-model", {"input": 1}, prices) is None


def test_a_model_without_a_cached_price_bills_cached_tokens_at_the_input_price():
    prices = {"judge": {"input": 1.0, "output": 5.0}}
    cost = tracing.cost_details("judge", {"input": 0, "input_cached_tokens": 1_000_000, "output": 0}, prices)
    assert cost["input_cached_tokens"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tracing_enabled": False},
        {"tracing_enabled": True, "langfuse_public_key": None, "langfuse_secret_key": None},  # keys missing
    ],
)
def test_tracing_stays_off_without_the_flag_and_both_keys(overrides):
    s = Settings(openai_api_key=SecretStr("sk-test"), model="m", **overrides)
    assert tracing._build(s) is tracing._OFF


def test_off_means_no_ops_and_no_trace_id(tmp_path):
    run_id = uuid.uuid4()
    assert tracing.client() is None
    assert tracing.trace_id_for(run_id) is None
    assert tracing.start_generation("x", "m") is tracing.NOOP
    with tracing.observe("x") as obs, tracing.run_trace(run_id, "agent", as_type="agent", input={}) as root:
        assert obs is tracing.NOOP and root is tracing.NOOP


def test_off_still_puts_cost_in_the_run_log(tmp_path):
    priced = _ask_settings(tmp_path).model_copy(update={"prices": {"test-model": {"input": 1.0, "output": 2.0}}})
    ask("What does Stripe do?", CHUNKS, client=_client(COMPLETED), settings=priced)
    (record,) = _read_runs(tmp_path)
    assert "trace_id" not in record  # not traced
    assert record["cost_usd"] > 0


# --- model calls ----------------------------------------------------------------------------


def test_a_model_call_is_a_generation_with_tokens_cost_and_its_trace_id_logged(spans, tmp_path):
    priced = _ask_settings(tmp_path).model_copy(update={"prices": {"test-model": {"input": 1.0, "output": 2.0}}})
    ask("What does Stripe do?", CHUNKS, client=_client(COMPLETED), settings=priced)

    (gen,) = spans()
    assert (gen["name"], gen["type"], gen["level"]) == ("ask", "generation", "DEFAULT")
    assert gen["attrs"]["langfuse.observation.model.name"] == "test-model"
    usage = json.loads(gen["attrs"]["langfuse.observation.usage_details"])
    assert usage["input"] + usage.get("input_cached_tokens", 0) == COMPLETED["usage"]["input_tokens"]
    assert "langfuse.observation.cost_details" in gen["attrs"]
    (record,) = _read_runs(tmp_path)
    assert record["trace_id"] == gen["trace_id"]  # the JSONL line points at its trace


def test_a_failed_model_call_is_an_error_generation(spans, tmp_path):
    truncated = {**COMPLETED, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    with pytest.raises(Exception, match="incomplete"):
        ask("q", CHUNKS, client=_client(truncated), settings=_ask_settings(tmp_path))
    (gen,) = spans()
    assert gen["level"] == "ERROR"
    assert "max_output_tokens" in gen["attrs"]["langfuse.observation.status_message"]


# --- an agent run: one trace, tool spans carry the gate ----------------------------------------


def test_an_agent_run_is_one_trace_with_the_gate_on_every_tool_span(spans, state_db, tmp_path):
    agent_ctx = tools.ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")
    agent_settings = Settings(openai_api_key=SecretStr("sk-test"), model="test-model", runs_dir=tmp_path / "runs")
    model = FakeModel(
        turn(call("get_lead", {"lead_id": "L1"}, "c1"), call("send_email", {"draft_id": "D1"}, "c2")),
        final("done"),
    )
    def deny_get_lead(c):
        return "deny" if c.name == "get_lead" else authorize(c)

    result = run_agent(
        "task",
        agent_ctx,
        respond_fn=model,
        authorize_fn=deny_get_lead,
        approve_fn=lambda c, x: False,
        settings=agent_settings,
    )

    obs = {o["name"]: o for o in spans()}
    trace_id = tracing.trace_id_for(result.run_id)
    assert {o["trace_id"] for o in obs.values()} == {trace_id}  # ONE trace, seeded from run_id
    assert state.get_trace_id(state_db, result.run_id) == trace_id  # ... and stored on the run
    assert {r["trace_id"] for r in _runs(agent_settings)} == {trace_id}  # ... and on every run-log line

    root = obs["agent"]
    assert (root["type"], root["attrs"]["langfuse.trace.name"]) == ("agent", "agent")
    assert json.loads(root["attrs"]["langfuse.trace.metadata.run_id"]) == str(result.run_id)

    denied, rejected = obs["get_lead"], obs["send_email"]
    assert (denied["type"], denied["parent"], denied["level"]) == ("tool", "agent", "WARNING")
    assert json.loads(denied["metadata"]["decision"]) == "deny"
    assert json.loads(denied["metadata"]["status"]) == "denied"
    assert json.loads(rejected["metadata"]["decision"]) == "needs_approval"
    assert json.loads(rejected["metadata"]["approved"]) is False
    assert obs["human-approval"]["parent"] == "send_email"  # human wait time is its own span


# --- a flow run: nodes, pause, approve -> still one trace -------------------------------------


def test_a_paused_then_approved_flow_is_one_trace(spans, ctx, settings, fake, saver):  # noqa: F811
    fake["verdicts"].append(STRONG)
    run_flow("L1", ctx, settings, checkpointer=saver)
    saver.threads.append(str(ctx.run_id))
    approve_flow("D1", True, ctx, saver)

    obs = spans()
    trace_id = tracing.trace_id_for(ctx.run_id)
    assert {o["trace_id"] for o in obs} == {trace_id}
    assert state.get_trace_id(ctx.conn, ctx.run_id) == trace_id
    roots = [o["name"] for o in obs if o["parent"] is None]
    assert roots == ["flow", "flow-approve"]  # two invocations, days apart maybe, one trace

    by_parent = {(o["parent"], o["name"]) for o in obs}
    assert {("flow", "load_lead"), ("flow", "research"), ("flow", "qualify"), ("flow", "draft")} <= by_parent
    assert {("load_lead", "lead_status"), ("qualify", "qualify_lead"), ("draft", "draft_email")} <= by_parent
    assert {("flow-approve", "approve"), ("flow-approve", "send"), ("send", "send_email")} <= by_parent
    assert all(o["level"] != "ERROR" for o in obs)  # interrupt() is a pause, not an error
    paused = next(o for o in obs if o["name"] == "flow")
    assert json.loads(paused["attrs"]["langfuse.observation.output"])["status"] == "waiting"


def test_a_permanent_node_failure_is_an_error_on_the_node_and_the_root(spans, ctx, settings, fake, monkeypatch):  # noqa: F811
    def broken_judge(*args, **kwargs):
        raise ValueError("judge exploded")

    monkeypatch.setattr(tools, "judge_icp", broken_judge)
    final_state = run_flow("L1", ctx, settings)
    assert final_state["error"]["node"] == "qualify"

    obs = {o["name"]: o for o in spans()}
    assert obs["qualify"]["level"] == "ERROR"
    assert obs["qualify_lead"]["level"] == "ERROR"  # the exception escaped the tool span first
    assert obs["flow"]["level"] == "ERROR"
    assert obs["failed"]["parent"] == "flow"


def test_call_tool_records_a_deny_before_refusing(spans, ctx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(flow, "authorize", lambda c: "deny")
    with pytest.raises(flow.FlowError):
        flow.call_tool(ctx, "get_lead", {"lead_id": "L1"})
    (tool,) = spans()
    assert json.loads(tool["metadata"]["decision"]) == "deny" and tool["level"] == "ERROR"
