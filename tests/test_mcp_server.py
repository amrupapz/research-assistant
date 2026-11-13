# ruff: noqa: F811  (asks/ctx are pytest fixtures imported from test_memory_pg and used by name)
"""Day 7 stretch: the MCP server, through the REAL protocol (the SDK's in-process Client → our
Server → mcp_call → the P3 gate → the test DB, rolled back). Fake model, no network."""

import anyio
import pytest
from mcp import Client
from test_agent import _settings
from test_memory_pg import CHUNK, SOLID, asks, ctx  # noqa: F401

from research_assistant import llm, mcp_server, state, tools
from research_assistant.db import seed_leads, set_tenant
from research_assistant.mcp_server import EXPOSED, build_server
from research_assistant.schemas import Lead
from research_assistant.tools import REGISTRY, tool_schema


def run(ctx, fn):
    """Connect a real MCP client to a server bound to ctx and run fn(client)."""

    async def go():
        async with Client(build_server(ctx)) as client:
            return await fn(client)

    return anyio.run(go)


def call(ctx, name, args):
    return run(ctx, lambda c: c.call_tool(name, args))


def count(ctx, table):
    return ctx.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --- listing --------------------------------------------------------------------------------


def test_exactly_the_three_read_only_tools_are_listed(ctx):
    names = [t.name for t in run(ctx, lambda c: c.list_tools()).tools]
    assert names == ["search_company", "get_lead", "qualify_lead"]
    assert "send_email" not in names and "draft_email" not in names


def test_each_schema_is_the_one_generated_from_its_pydantic_model(ctx):
    for t in run(ctx, lambda c: c.list_tools()).tools:
        assert t.input_schema == tool_schema(REGISTRY[t.name])["parameters"]
        assert t.input_schema["additionalProperties"] is False
        assert t.description == REGISTRY[t.name].description


# --- calls ----------------------------------------------------------------------------------


def test_get_lead_returns_structured_data(ctx):
    r = call(ctx, "get_lead", {"lead_id": "L1"})
    assert not r.is_error
    assert r.structured_content["company"] == "Figma" and r.structured_content["name"] == "Dana Kim"


def test_search_company_goes_through_the_gate_and_returns_cited_research(ctx, asks):
    r = call(ctx, "search_company", {"company": "Figma", "question": "What does Figma do?"})
    assert not r.is_error and asks == ["What does Figma do?"]
    assert r.structured_content["citations"][0]["chunk_id"] == "figma-about#000"


def test_another_tenants_lead_does_not_exist_here(ctx):
    ctx.conn.execute("INSERT INTO tenants (tenant_id) VALUES ('acme') ON CONFLICT DO NOTHING")
    set_tenant(ctx.conn, "acme")
    seed_leads(ctx.conn, [Lead(lead_id="L9", name="Sam Roe", title="CTO", company="Figma", email="sam@example.com")])
    set_tenant(ctx.conn, "default")
    r = call(ctx, "get_lead", {"lead_id": "L9"})
    assert r.is_error and "no lead" in r.content[0].text


# --- what must stay impossible ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["send_email", "draft_email", "lead_status", "Get_Lead", ""])
def test_unexposed_tools_are_refused_and_nothing_happens(ctx, name):
    r = call(ctx, name, {"draft_id": "D1", "lead_id": "L1", "subject": "s", "body": "b"})
    assert r.is_error and "unknown tool" in r.content[0].text
    assert (count(ctx, "drafts"), count(ctx, "sends"), count(ctx, "approvals")) == (0, 0, 0)


def test_even_if_send_email_were_exposed_the_gate_still_refuses_it(ctx, monkeypatch):
    """Defense in depth: exposure is one check, authorize() inside call_tool is another."""
    monkeypatch.setattr(mcp_server, "EXPOSED", (*EXPOSED, "send_email"))
    r = call(ctx, "send_email", {"draft_id": "D1"})
    assert r.is_error and "decision=needs_approval" in r.content[0].text
    assert count(ctx, "sends") == 0


@pytest.mark.parametrize(
    "args",
    [{"lead_id": "L1; DROP TABLE leads"}, {"lead_id": "L1", "admin": True}, {}],
)
def test_invalid_arguments_are_refused_without_echoing_them(ctx, args):
    r = call(ctx, "get_lead", args)
    assert r.is_error and "invalid arguments for get_lead" in r.content[0].text
    assert "DROP TABLE" not in r.content[0].text


# --- money ------------------------------------------------------------------------------------


def test_over_budget_model_tools_are_refused_before_the_model(ctx, monkeypatch):
    ctx.conn.execute("UPDATE tenants SET daily_token_budget = 100 WHERE tenant_id = 'default'")
    state.record_usage(ctx.conn, 100, "ask")
    monkeypatch.setattr(tools, "hybrid_search", lambda *a, **k: pytest.fail("embedding call after budget refusal"))
    monkeypatch.setattr(tools, "ask", lambda *a, **k: pytest.fail("model call after budget refusal"))
    r = call(ctx, "search_company", {"company": "Figma", "question": "What does Figma do?"})
    assert r.is_error and "daily token budget exceeded (used 100 of 100" in r.content[0].text
    assert not call(ctx, "get_lead", {"lead_id": "L1"}).is_error  # no model call: still allowed


def test_model_tools_record_their_tokens(ctx, monkeypatch, tmp_path):
    settings = _settings(tmp_path / "runs")

    def metered_ask(question, chunks):
        llm._log({"input_tokens": 2_000, "output_tokens": 120}, settings)
        return SOLID

    monkeypatch.setattr(tools, "hybrid_search", lambda conn, q, k: [(CHUNK, 1.0)])
    monkeypatch.setattr(tools, "ask", metered_ask)
    call(ctx, "search_company", {"company": "Figma", "question": "What does Figma do?"})
    assert ctx.conn.execute("SELECT tenant_id, endpoint, tokens FROM usage").fetchall() == [
        ("default", "mcp:search_company", 2_120)
    ]


def test_an_unexpected_failure_says_what_not_how(ctx, monkeypatch, tmp_path):
    settings = _settings(tmp_path / "runs")
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)

    def boom(*a, **k):
        raise RuntimeError("secret detail sk-live-123 at /Users/me/code")

    monkeypatch.setattr(tools, "hybrid_search", boom)
    r = call(ctx, "search_company", {"company": "Figma", "question": "What does Figma do?"})
    assert r.is_error and r.content[0].text == "search_company failed (RuntimeError); details are in the server's run log"
    logged = (settings.runs_dir).glob("*.jsonl")
    assert any("secret detail" in f.read_text() for f in logged)  # the detail is in OUR log only
