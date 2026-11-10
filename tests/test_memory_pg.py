"""D4 M6: memory across days. Reuse fresh research/qualifications; know who was emailed, who
replied, and who is due a follow-up. "Days ago" is simulated by backdating rows."""

import json
from datetime import date

import pytest

from research_assistant import state, tools
from research_assistant.prompts import PROMPT_VERSION
from research_assistant.schemas import (
    Chunk,
    Citation,
    CriterionResult,
    IcpJudgment,
    ResearchAnswer,
)
from research_assistant.sending import send_once
from research_assistant.tools import ToolContext, run_tool

CHUNK = Chunk(
    chunk_id="figma-about#000",
    source_url="https://figma.com/about",
    fetched_at=date(2026, 11, 1),
    title="About Figma",
    text="Figma helps teams design.",
)
SOLID = ResearchAnswer(
    answer="A design tool [figma-about#000]",
    confidence="high",
    assumptions=["x"],
    needs_sources=False,
    citations=[
        Citation(chunk_id="figma-about#000", source_url="https://figma.com/about")
    ],
)


@pytest.fixture
def ctx(state_db, tmp_path):
    return ToolContext(conn=state_db, outbox_dir=tmp_path / "outbox")


@pytest.fixture
def asks(monkeypatch):
    """Fake research backend; returns the list of questions that really hit the model."""
    calls = []
    monkeypatch.setattr(tools, "hybrid_search", lambda conn, q, k: [(CHUNK, 1.0)])
    monkeypatch.setattr(tools, "ask", lambda q, chunks: calls.append(q) or SOLID)
    return calls


def _search(ctx, company="Figma", question="What does Figma do?"):
    return run_tool(
        "search_company", json.dumps({"company": company, "question": question}), ctx
    )


def _age(ctx, table, days, where="TRUE"):
    ctx.conn.execute(
        f"UPDATE {table} SET created_at = now() - make_interval(days => %s) WHERE {where}",
        [days],
    )


# --- research reuse -------------------------------------------------------------------------


def test_new_answer_is_saved_with_citation_snapshots(ctx, asks):
    _search(ctx)
    row = ctx.conn.execute(
        "SELECT company, question, confidence, prompt_version, assumptions FROM research_findings"
    ).fetchone()
    assert row == ("Figma", "What does Figma do?", "high", PROMPT_VERSION, ["x"])
    cit = ctx.conn.execute(
        "SELECT chunk_id, source_url, fetched_at FROM finding_citations"
    ).fetchone()
    assert cit == ("figma-about#000", "https://figma.com/about", date(2026, 11, 1))


def test_same_question_within_7_days_is_reused_without_a_model_call(ctx, asks):
    first = _search(ctx)
    again = _search(
        ctx, company="figma", question="  what does FIGMA do? "
    )  # case/space don't matter
    assert len(asks) == 1
    assert again["reused_from"] is not None
    assert {k: again[k] for k in first} == first  # the same answer, citations included


@pytest.mark.parametrize(
    ("setup", "company", "question"),
    [
        (
            lambda c: _age(c, "research_findings", 8),
            "Figma",
            "What does Figma do?",
        ),  # too old
        (lambda c: None, "Linear", "What does Figma do?"),  # other company
        (lambda c: None, "Figma", "Who is Figma's CEO?"),  # other question
        (
            lambda c: c.conn.execute(
                "UPDATE research_findings SET prompt_version = 'v3'"
            ),
            "Figma",
            "What does Figma do?",
        ),
        (
            lambda c: c.conn.execute(
                "UPDATE research_findings SET needs_sources = true"
            ),
            "Figma",
            "What does Figma do?",
        ),
    ],
    ids=[
        "older_than_7_days",
        "other_company",
        "other_question",
        "old_prompt",
        "shaky_answer",
    ],
)
def test_not_reused(ctx, asks, setup, company, question):
    _search(ctx)
    setup(ctx)
    out = _search(ctx, company, question)
    assert len(asks) == 2 and "reused_from" not in out


# --- qualification reuse --------------------------------------------------------------------


@pytest.fixture
def judges(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "hybrid_search", lambda conn, q, k: [(CHUNK, 1.0)])

    def judge(company, criteria, chunks):
        calls.append(company)
        return IcpJudgment(
            results=[
                CriterionResult(
                    id=cid, met="yes", reason="r", chunk_ids=["figma-about#000"]
                )
                for cid, _ in criteria
            ]
        )

    monkeypatch.setattr(tools, "judge_icp", judge)
    return calls


def test_qualification_within_7_days_is_reused(ctx, judges):
    first = run_tool("qualify_lead", '{"lead_id": "L1"}', ctx)
    again = run_tool("qualify_lead", '{"lead_id": "L1"}', ctx)
    assert len(judges) == 1
    assert (again["fit"], again["criteria"]) == (
        first["fit"],
        [c for c in first["criteria"]],
    )
    assert again["reused_from"] is not None
    assert set(again) == set(first) | {"reused_from"}  # same shape as a fresh result, plus the date


def test_old_or_other_icp_qualification_is_redone(ctx, judges):
    run_tool("qualify_lead", '{"lead_id": "L1"}', ctx)
    _age(ctx, "qualifications", 8)
    run_tool("qualify_lead", '{"lead_id": "L1"}', ctx)
    ctx.conn.execute("UPDATE qualifications SET icp_version = 'icp-v0'")
    run_tool("qualify_lead", '{"lead_id": "L1"}', ctx)
    assert len(judges) == 3


# --- outreach status + follow-ups -----------------------------------------------------------


def _sent(ctx, lead_id="L1", days_ago=0):
    """A real approved + sent email to the lead, then backdated."""
    lead = tools.fetch_lead(ctx.conn, lead_id)
    d = state.insert_draft(ctx.conn, lead, "Hi", "Hello", None)
    state.record_approval(ctx.conn, d.draft_id, True, None)
    send_once(ctx.conn, d, ctx.outbox_dir)
    ctx.conn.execute(
        "UPDATE sends SET sent_at = now() - make_interval(days => %s) WHERE draft_id = %s",
        [days_ago, int(d.draft_id[1:])],
    )
    return d


def _reply(ctx, lead_id="L1", days_ago=0):
    state.record_reply(ctx.conn, lead_id)
    ctx.conn.execute(
        "UPDATE replies SET received_at = now() - make_interval(days => %s) "
        "WHERE reply_id = (SELECT max(reply_id) FROM replies)",
        [days_ago],
    )


def _due(ctx, days=3):
    return [
        f["lead_id"]
        for f in run_tool("list_followups", json.dumps({"days": days}), ctx)["leads"]
    ]


def test_never_emailed_is_not_due(ctx):
    assert _due(ctx) == []


def test_emailed_4_days_ago_no_reply_is_due(ctx):
    _sent(ctx, "L1", days_ago=4)
    assert _due(ctx) == ["L1"]


def test_emailed_2_days_ago_is_not_due_yet(ctx):
    _sent(ctx, "L1", days_ago=2)
    assert _due(ctx) == []
    assert _due(ctx, days=1) == ["L1"]


def test_reply_after_our_email_means_not_due(ctx):
    _sent(ctx, "L1", days_ago=4)
    _reply(ctx, "L1", days_ago=1)
    assert _due(ctx) == []


def test_reply_before_our_latest_email_does_not_count(ctx):
    _reply(ctx, "L1", days_ago=10)
    _sent(ctx, "L1", days_ago=4)
    assert _due(ctx) == ["L1"]


def test_a_follow_up_resets_the_clock(ctx):
    _sent(ctx, "L1", days_ago=6)
    _sent(ctx, "L1", days_ago=1)  # the follow-up
    assert _due(ctx) == []
    assert state.followups_due(ctx.conn, 0)[0]["emails_sent"] == 2


def test_unsent_drafts_are_not_contact(ctx):
    state.insert_draft(ctx.conn, tools.fetch_lead(ctx.conn, "L1"), "Hi", "Hello", None)
    ctx.conn.execute("UPDATE drafts SET created_at = now() - interval '10 days'")
    assert _due(ctx) == []


def test_oldest_contact_first(ctx):
    _sent(ctx, "L3", days_ago=5)
    _sent(ctx, "L1", days_ago=9)
    assert _due(ctx) == ["L1", "L3"]


@pytest.mark.parametrize("days", [0, 31, -1])
def test_days_is_bounded(ctx, days):
    with pytest.raises(tools.ToolError, match="days"):
        run_tool("list_followups", json.dumps({"days": days}), ctx)


def test_lead_status_is_the_full_memory(ctx, asks):
    d = _sent(ctx, "L1", days_ago=4)
    state.insert_draft(ctx.conn, tools.fetch_lead(ctx.conn, "L1"), "Re", "Unsent", None)
    _search(ctx)
    status = run_tool("lead_status", '{"lead_id": "L1"}', ctx)
    o = status["outreach"]
    assert [s["draft_id"] for s in o["sent"]] == [d.draft_id]
    assert o["drafts_unsent"] == ["D2"]
    assert (
        o["last_contacted_at"] is not None and o["replied_since_last_contact"] is False
    )
    assert [r["question"] for r in status["recent_research"]] == ["What does Figma do?"]
    assert status["lead"]["company"] == "Figma"
