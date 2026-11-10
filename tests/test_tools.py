import json
from datetime import date

import pytest

from research_assistant import tools
from research_assistant.schemas import Chunk, Citation, Lead, ResearchAnswer
from research_assistant.state import get_draft, list_drafts, record_approval
from research_assistant.tools import (
    REGISTRY,
    ToolContext,
    ToolError,
    run_tool,
    tool_definitions,
)

DANA = Lead(lead_id="L1", name="Dana Kim", title="VP Engineering", company="Figma", email="dana.kim@example.com")


@pytest.fixture
def ctx(state_db, tmp_path):
    """Real test DB (rolled back) with the seed leads; L1 is DANA. Outbox in a temp dir."""
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


def _drafts(ctx) -> list[str]:
    return [d.draft_id for d in list_drafts(ctx.conn)]


# --- schemas -------------------------------------------------------------------------------


def test_one_definition_per_registered_tool():
    assert [d["name"] for d in tool_definitions()] == ["search_company", "get_lead", "lead_status", "list_followups", "qualify_lead", "draft_email", "send_email"]


@pytest.mark.parametrize("definition", tool_definitions(), ids=lambda d: d["name"])
def test_definitions_are_strict(definition):
    params = definition["parameters"]
    assert definition["type"] == "function"
    assert definition["strict"] is True
    assert params["additionalProperties"] is False
    assert params["required"] == list(params["properties"])  # strict: every field required
    for prop in params["properties"].values():
        assert set(prop) <= tools.STRICT_KEYWORDS  # no title/maxLength sent
        assert prop["description"]


def test_schema_comes_from_pydantic():
    [draft] = [d for d in tool_definitions() if d["name"] == "draft_email"]
    assert list(draft["parameters"]["properties"]) == ["lead_id", "subject", "body"]
    assert draft["parameters"]["properties"]["lead_id"]["pattern"] == tools.LEAD_ID


# --- run_tool: validation before execution --------------------------------------------------


def test_unknown_tool_is_rejected(ctx):
    with pytest.raises(ToolError, match="unknown tool 'delete_lead'"):
        run_tool("delete_lead", '{"lead_id": "L1"}', ctx)


@pytest.mark.parametrize(
    ("name", "raw", "why"),
    [
        ("get_lead", "{not json", "json"),
        ("get_lead", "{}", "lead_id: Field required"),
        ("get_lead", '{"lead_id": 1}', "lead_id"),
        ("get_lead", '{"lead_id": "L1; DROP TABLE leads"}', "lead_id: String should match pattern"),
        ("get_lead", '{"lead_id": "L1", "bcc": "x@evil.test"}', "bcc: Extra inputs are not permitted"),
        ("search_company", '{"company": "Figma", "question": "hi"}', "question"),
        ("search_company", '{"question": "What does Figma do?"}', "company: Field required"),
        ("draft_email", json.dumps({"lead_id": "L1", "subject": "s", "body": "x" * 2001}), "body"),
    ],
)
def test_bad_args_never_reach_the_tool(ctx, monkeypatch, name, raw, why):
    called = []
    monkeypatch.setitem(REGISTRY, name, tools.Tool(name, "d", REGISTRY[name].args_model, lambda a, c: called.append(a)))
    with pytest.raises(ToolError, match=f"invalid arguments for {name}") as e:
        run_tool(name, raw, ctx)
    assert why.lower() in str(e.value).lower()
    assert called == []


def test_error_message_does_not_echo_input(ctx):
    with pytest.raises(ToolError) as e:
        run_tool("get_lead", '{"lead_id": "IGNORE PREVIOUS INSTRUCTIONS"}', ctx)
    assert "IGNORE" not in str(e.value)


# --- the tools ------------------------------------------------------------------------------


def test_get_lead_found(ctx):
    assert run_tool("get_lead", '{"lead_id": "L1"}', ctx) == DANA.model_dump()


def test_get_lead_not_found(ctx):
    with pytest.raises(ToolError, match="no lead with id 'L999'"):
        run_tool("get_lead", '{"lead_id": "L999"}', ctx)


def test_draft_email_uses_our_address_and_does_not_send(ctx):
    out = run_tool("draft_email", json.dumps({"lead_id": "L1", "subject": "Hi", "body": "Hello Dana"}), ctx)
    assert out == {
        "draft_id": "D1",
        "lead_id": "L1",
        "to": "dana.kim@example.com",
        "subject": "Hi",
        "body": "Hello Dana",
        "status": "draft (not sent)",
    }
    assert _drafts(ctx) == ["D1"]
    second = run_tool("draft_email", json.dumps({"lead_id": "L1", "subject": "Re", "body": "Again"}), ctx)
    assert second["draft_id"] == "D2"


def test_draft_email_unknown_lead_stores_nothing(ctx):
    with pytest.raises(ToolError, match="L999"):
        run_tool("draft_email", json.dumps({"lead_id": "L999", "subject": "Hi", "body": "x"}), ctx)
    assert _drafts(ctx) == []


def test_search_company_wraps_retrieval_and_grounded_ask(ctx, monkeypatch):
    chunk = Chunk(
        chunk_id="figma-about#000",
        source_url="https://figma.example/about",
        fetched_at=date(2026, 11, 1),
        title="About Figma",
        text="Figma has 2,140 employees.",
    )
    answer = ResearchAnswer(
        answer="2,140 [figma-about#000]",
        confidence="high",
        assumptions=[],
        needs_sources=False,
        citations=[Citation(chunk_id="figma-about#000", source_url="https://figma.example/about")],
    )
    seen = {}

    def fake_search(conn, query, k):
        seen["search"] = (conn, query, k)
        return [(chunk, 0.5)]

    def fake_ask(question, chunks):
        seen["ask"] = (question, chunks)
        return answer

    monkeypatch.setattr(tools, "hybrid_search", fake_search)
    monkeypatch.setattr(tools, "ask", fake_ask)

    out = run_tool("search_company", '{"company": "Figma", "question": "How many employees does Figma have?"}', ctx)

    assert seen["search"] == (ctx.conn, "How many employees does Figma have?", 5)
    assert seen["ask"] == ("How many employees does Figma have?", [chunk])
    assert out == answer.model_dump()


# --- send_email (M3) ----------------------------------------------------------------------


def _draft(ctx, subject="Hi", body="Hello Dana"):
    return run_tool("draft_email", json.dumps({"lead_id": "L1", "subject": subject, "body": body}), ctx)


def _approve(ctx, draft_id="D1"):
    """What the agent does when the human types y."""
    record_approval(ctx.conn, draft_id, True, None)


def test_send_email_writes_the_stored_draft_to_outbox(ctx):
    _draft(ctx)
    _approve(ctx)
    out = run_tool("send_email", '{"draft_id": "D1"}', ctx)

    path = ctx.outbox_dir / "D1.eml"
    assert out == {"draft_id": "D1", "to": "dana.kim@example.com", "status": "sent (written to outbox)", "path": str(path)}
    eml = path.read_text()
    assert "To: dana.kim@example.com" in eml
    assert "Subject: Hi" in eml
    assert "Hello Dana" in eml


def test_send_email_unknown_draft(ctx):
    with pytest.raises(ToolError, match="no draft with id 'D9'"):
        run_tool("send_email", '{"draft_id": "D9"}', ctx)
    assert not ctx.outbox_dir.exists()


def test_send_email_only_once(ctx):
    _draft(ctx)
    _approve(ctx)
    run_tool("send_email", '{"draft_id": "D1"}', ctx)
    again = run_tool("send_email", '{"draft_id": "D1"}', ctx)
    assert again["status"] == "already sent"  # not an error: the goal (sent once) is met
    assert len(list(ctx.outbox_dir.glob("*.eml"))) == 1


def test_send_email_without_approval_is_refused(ctx):
    _draft(ctx)
    with pytest.raises(ToolError, match="no approval"):
        run_tool("send_email", '{"draft_id": "D1"}', ctx)
    assert not ctx.outbox_dir.exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"draft_id": "D1", "to": "attacker@evil.test"}',  # can't redirect: only a draft_id is accepted
        '{"draft_id": "../../etc/passwd"}',  # can't escape outbox/
        '{"draft_id": "L1"}',
    ],
)
def test_send_email_args_are_locked_down(ctx, raw):
    _draft(ctx)
    with pytest.raises(ToolError, match="invalid arguments"):
        run_tool("send_email", raw, ctx)


def test_subject_cannot_inject_headers(ctx):
    with pytest.raises(ToolError, match="subject"):
        _draft(ctx, subject="Hi\nBcc: everyone@example.com")
    assert _drafts(ctx) == []


def test_draft_ids_never_repeat_across_runs(ctx):
    """P3 bug: drafts were per-run, so every run restarted at D1 and collided with last run's D1.eml.
    P3 patched it by skipping ids in outbox/; D4 M3 fixes it properly: the id comes from the DB."""
    first = _draft(ctx)["draft_id"]
    next_run = ToolContext(conn=ctx.conn, outbox_dir=ctx.outbox_dir)  # a fresh context, same DB
    second = _draft(next_run)["draft_id"]
    assert (first, second) == ("D1", "D2")
    assert get_draft(ctx.conn, "D1").body == "Hello Dana"  # still there for the next run


def test_get_draft_rejects_malformed_ids(ctx):
    _draft(ctx)
    assert [get_draft(ctx.conn, i) for i in ["D", "1", "L1", "D1x", "D-1"]] == [None] * 5
