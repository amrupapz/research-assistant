"""Agent state in Postgres (D4 M3): runs, steps, drafts, approvals, qualifications.

Plain SQL, no commits: the CALLER owns the transaction. agent.py wraps each step and its effect
in one `with conn.transaction():` so they commit together or not at all.

Tenants (D7 M2): every SELECT/UPDATE filters on `tenant_id = {TENANT}` (the connection's tenant,
db.set_tenant); every INSERT gets it from the column default. Another tenant's row = no row.
"""

import json
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from research_assistant.db import TENANT
from research_assistant.schemas import Draft, Lead, Qualification


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=lambda v: json.dumps(v, default=str))  # dates etc. -> strings


# --- runs + steps ---------------------------------------------------------------------------


def start_run(conn: psycopg.Connection, task: str, model: str, agent_prompt_version: str) -> UUID:
    return conn.execute(
        "INSERT INTO runs (task, model, agent_prompt_version) VALUES (%s, %s, %s) RETURNING run_id",
        [task, model, agent_prompt_version],
    ).fetchone()[0]


def set_trace_id(conn: psycopg.Connection, run_id: UUID, trace_id: str | None) -> None:
    """Link the run to its Langfuse trace (D6 M1). No-op when tracing is off (trace_id None)."""
    if trace_id is not None:
        conn.execute(f"UPDATE runs SET trace_id = %s WHERE run_id = %s AND tenant_id = {TENANT}", [trace_id, run_id])


def get_trace_id(conn: psycopg.Connection, run_id: UUID) -> str | None:
    row = conn.execute(f"SELECT trace_id FROM runs WHERE run_id = %s AND tenant_id = {TENANT}", [run_id]).fetchone()
    return row[0] if row else None


def get_run(conn: psycopg.Connection, run_id: UUID) -> dict | None:
    row = conn.execute(
        f"SELECT task, status, model, agent_prompt_version FROM runs WHERE run_id = %s AND tenant_id = {TENANT}", [run_id]
    ).fetchone()
    return dict(zip(["task", "status", "model", "agent_prompt_version"], row, strict=True)) if row else None


def mark_running(conn: psycopg.Connection, run_id: UUID) -> None:
    conn.execute(f"UPDATE runs SET status = 'running', finished_at = NULL WHERE run_id = %s AND tenant_id = {TENANT}", [run_id])


def claim_waiting(conn: psycopg.Connection, run_id: UUID) -> bool:
    """waiting → running only if it is STILL waiting; True if this call won (exactly one caller does)."""
    cur = conn.execute(f"UPDATE runs SET status = 'running' WHERE run_id = %s AND status = 'waiting' AND tenant_id = {TENANT}", [run_id])
    return cur.rowcount == 1


def wait_run(conn: psycopg.Connection, run_id: UUID, answer: str | None = None) -> None:
    """Paused for a human (a flow at interrupt()): not finished, so finished_at stays NULL."""
    conn.execute(f"UPDATE runs SET status = 'waiting', answer = %s, finished_at = NULL WHERE run_id = %s AND tenant_id = {TENANT}", [answer, run_id])


def finish_run(conn: psycopg.Connection, run_id: UUID, status: str, answer: str | None = None) -> None:
    conn.execute(
        f"UPDATE runs SET status = %s, answer = %s, finished_at = now() WHERE run_id = %s AND tenant_id = {TENANT}",
        [status, answer, run_id],
    )


def add_step(
    conn: psycopg.Connection,
    run_id: UUID,
    *,
    turn: int,
    kind: str,
    payload: Any,
    call_id: str | None = None,
    tool: str | None = None,
    decision: str | None = None,
    approved: bool | None = None,
    status: str | None = None,
    output: Any = None,
) -> int:
    """Append a step; seq = last seq + 1 (read from the DB, so a resumed run continues the numbering)."""
    return conn.execute(
        """
        INSERT INTO run_steps (run_id, seq, turn, kind, call_id, tool, payload, decision, approved, status, output)
        VALUES (%(run_id)s,
                (SELECT coalesce(max(seq), 0) + 1 FROM run_steps WHERE run_id = %(run_id)s),
                %(turn)s, %(kind)s, %(call_id)s, %(tool)s, %(payload)s, %(decision)s, %(approved)s, %(status)s, %(output)s)
        RETURNING seq
        """,
        {
            "run_id": run_id,
            "turn": turn,
            "kind": kind,
            "call_id": call_id,
            "tool": tool,
            "payload": _jsonb(payload),
            "decision": decision,
            "approved": approved,
            "status": status,
            "output": None if output is None else _jsonb(output),
        },
    ).fetchone()[0]


def get_steps(conn: psycopg.Connection, run_id: UUID) -> list[dict]:
    cur = conn.execute(
        "SELECT seq, turn, kind, call_id, tool, payload, decision, approved, status, output "
        f"FROM run_steps WHERE run_id = %s AND tenant_id = {TENANT} ORDER BY seq",
        [run_id],
    )
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


# --- drafts + approvals ---------------------------------------------------------------------
# The DB id is an integer identity; the model and the human see "D<id>".


def _draft_number(draft_id: str) -> int | None:
    return int(draft_id[1:]) if draft_id[:1] == "D" and draft_id[1:].isdigit() else None


def _draft(row) -> Draft:
    n, lead_id, to_email, subject, body = row
    return Draft(draft_id=f"D{n}", lead_id=lead_id, to=to_email, subject=subject, body=body)


def insert_draft(conn: psycopg.Connection, lead: Lead, subject: str, body: str, run_id: UUID | None) -> Draft:
    """to_email is copied from the lead NOW: what you approve is what gets sent, even if the lead changes."""
    row = conn.execute(
        "INSERT INTO drafts (lead_id, to_email, subject, body, run_id) VALUES (%s, %s, %s, %s, %s) "
        "RETURNING draft_id, lead_id, to_email, subject, body",
        [lead.lead_id, lead.email, subject, body, run_id],
    ).fetchone()
    return _draft(row)


def get_draft(conn: psycopg.Connection, draft_id: str) -> Draft | None:
    n = _draft_number(draft_id)
    if n is None:
        return None
    row = conn.execute(
        f"SELECT draft_id, lead_id, to_email, subject, body FROM drafts WHERE draft_id = %s AND tenant_id = {TENANT}", [n]
    ).fetchone()
    return _draft(row) if row else None


def draft_run_id(conn: psycopg.Connection, draft_id: str) -> UUID | None:
    """The run that wrote this draft (None if no such draft, or it was written outside a run)."""
    n = _draft_number(draft_id)
    row = conn.execute(f"SELECT run_id FROM drafts WHERE draft_id = %s AND tenant_id = {TENANT}", [n]).fetchone() if n else None
    return row[0] if row else None


def list_drafts(conn: psycopg.Connection, run_id: UUID | None = None) -> list[Draft]:
    """All drafts (or one run's), oldest first."""
    sql = f"SELECT draft_id, lead_id, to_email, subject, body FROM drafts WHERE tenant_id = {TENANT}"
    rows = conn.execute(sql + " AND run_id = %s ORDER BY draft_id", [run_id]).fetchall() if run_id else (
        conn.execute(sql + " ORDER BY draft_id").fetchall()
    )
    return [_draft(r) for r in rows]


def record_approval(conn: psycopg.Connection, draft_id: str, approved: bool, run_id: UUID | None) -> int | None:
    """Record the human's y/N for a draft. Returns approval_id, or None if there's no such draft."""
    n = _draft_number(draft_id)
    if n is None:
        return None
    row = conn.execute(
        "INSERT INTO approvals (draft_id, decision, run_id) "
        f"SELECT draft_id, %s, %s FROM drafts WHERE draft_id = %s AND tenant_id = {TENANT} "  # 0 rows if no draft
        "RETURNING approval_id",
        ["approved" if approved else "declined", run_id, n],
    ).fetchone()
    return row[0] if row else None


# --- qualifications -------------------------------------------------------------------------


def save_qualification(conn: psycopg.Connection, q: Qualification, run_id: UUID | None) -> int:
    return conn.execute(
        "INSERT INTO qualifications (lead_id, fit, criteria, icp_version, run_id) VALUES (%s, %s, %s, %s, %s) "
        "RETURNING qualification_id",
        [q.lead_id, q.fit, _jsonb([c.model_dump() for c in q.criteria]), q.icp_version, run_id],
    ).fetchone()[0]


def latest_qualification(conn: psycopg.Connection, lead_id: str) -> dict | None:
    row = conn.execute(
        f"SELECT fit, criteria, icp_version, created_at FROM qualifications WHERE lead_id = %s AND tenant_id = {TENANT} "
        "ORDER BY created_at DESC, qualification_id DESC LIMIT 1",
        [lead_id],
    ).fetchone()
    return dict(zip(["fit", "criteria", "icp_version", "created_at"], row, strict=True)) if row else None


# --- sends: the ledger that makes "send exactly once" possible (D4 M5) ----------------------


def latest_approval(conn: psycopg.Connection, draft_id: str) -> tuple[int, str] | None:
    """(approval_id, 'approved' | 'declined') of the human's MOST RECENT answer for this draft, or None."""
    n = _draft_number(draft_id)
    if n is None:
        return None
    return conn.execute(
        f"SELECT approval_id, decision FROM approvals WHERE draft_id = %s AND tenant_id = {TENANT} "
        "ORDER BY approval_id DESC LIMIT 1",
        [n],
    ).fetchone()


def claim_send(conn: psycopg.Connection, draft_id: str, approval_id: int, key: str, channel: str) -> int | None:
    """Write 'about to send' (status pending). Returns send_id, or None if this draft is ALREADY claimed.

    ON CONFLICT DO NOTHING: if the UNIQUE(draft_id) or UNIQUE(idempotency_key) row exists, insert
    nothing and return no row (instead of raising and aborting the transaction).
    """
    row = conn.execute(
        "INSERT INTO sends (draft_id, approval_id, idempotency_key, channel) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING RETURNING send_id",
        [_draft_number(draft_id), approval_id, key, channel],
    ).fetchone()
    return row[0] if row else None


def get_send(conn: psycopg.Connection, draft_id: str) -> dict | None:
    row = conn.execute(
        "SELECT send_id, status, idempotency_key, channel, sent_at, provider_message_id FROM sends "
        f"WHERE draft_id = %s AND tenant_id = {TENANT}",
        [_draft_number(draft_id)],
    ).fetchone()
    cols = ["send_id", "status", "idempotency_key", "channel", "sent_at", "provider_message_id"]
    return dict(zip(cols, row, strict=True)) if row else None


def mark_sent(conn: psycopg.Connection, draft_id: str, provider_message_id: str) -> None:
    conn.execute(
        "UPDATE sends SET status = 'sent', sent_at = now(), provider_message_id = %s "
        f"WHERE draft_id = %s AND tenant_id = {TENANT}",
        [provider_message_id, _draft_number(draft_id)],
    )


# --- memory across days (D4 M6) -------------------------------------------------------------
# All "is it fresh / did they reply / who's waiting" decisions are SQL over exact facts.

REUSE_DAYS = 7  # research and qualifications younger than this are reused, not re-run


def save_finding(
    conn: psycopg.Connection,
    company: str,
    question: str,
    answer: dict,
    fetched_at_by_chunk: dict[str, Any],
    prompt_version: str,
    run_id: UUID | None,
) -> int:
    """Store a search_company answer + its citations (snapshots: chunk_id, source_url, fetched_at)."""
    finding_id = conn.execute(
        "INSERT INTO research_findings (company, question, answer, confidence, needs_sources, assumptions, "
        "prompt_version, run_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING finding_id",
        [company, question, answer["answer"], answer["confidence"], answer["needs_sources"],
         _jsonb(answer["assumptions"]), prompt_version, run_id],
    ).fetchone()[0]
    for c in answer["citations"]:
        conn.execute(
            "INSERT INTO finding_citations VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [finding_id, c["chunk_id"], c["source_url"], fetched_at_by_chunk[c["chunk_id"]]],
        )
    return finding_id


def reusable_finding(conn: psycopg.Connection, company: str, question: str, prompt_version: str) -> dict | None:
    """The newest SOLID answer to the same question about the same company, < REUSE_DAYS old.

    Same question = equal after lowercasing + trimming. Shaky answers (needs_sources) and answers
    from another prompt version are never reused: those deserve a fresh look.
    """
    row = conn.execute(
        f"""
        SELECT finding_id, answer, confidence, needs_sources, assumptions, created_at FROM research_findings
        WHERE tenant_id = {TENANT} AND lower(company) = lower(%s) AND lower(btrim(question)) = lower(btrim(%s))
          AND prompt_version = %s AND NOT needs_sources
          AND created_at > now() - make_interval(days => %s)
        ORDER BY created_at DESC LIMIT 1
        """,
        [company, question, prompt_version, REUSE_DAYS],
    ).fetchone()
    if row is None:
        return None
    finding_id, answer, confidence, needs_sources, assumptions, created_at = row
    citations = conn.execute(
        "SELECT chunk_id, source_url FROM finding_citations WHERE finding_id = %s ORDER BY chunk_id", [finding_id]
    ).fetchall()
    return {
        "answer": answer,
        "confidence": confidence,
        "assumptions": assumptions,
        "needs_sources": needs_sources,
        "citations": [{"chunk_id": c, "source_url": u} for c, u in citations],
        "reused_from": created_at,
    }


def recent_findings(conn: psycopg.Connection, company: str) -> list[dict]:
    rows = conn.execute(
        "SELECT question, confidence, needs_sources, created_at FROM research_findings "
        f"WHERE tenant_id = {TENANT} AND lower(company) = lower(%s) AND created_at > now() - make_interval(days => %s) ORDER BY created_at DESC",
        [company, REUSE_DAYS],
    ).fetchall()
    return [dict(zip(["question", "confidence", "needs_sources", "created_at"], r, strict=True)) for r in rows]


def reusable_qualification(conn: psycopg.Connection, lead_id: str, icp_version: str) -> dict | None:
    """Latest qualification for this lead under the SAME ICP version, < REUSE_DAYS old."""
    row = conn.execute(
        f"SELECT fit, criteria, created_at FROM qualifications WHERE tenant_id = {TENANT} AND lead_id = %s "
        "AND icp_version = %s "
        "AND created_at > now() - make_interval(days => %s) ORDER BY created_at DESC, qualification_id DESC LIMIT 1",
        [lead_id, icp_version, REUSE_DAYS],
    ).fetchone()
    return dict(zip(["fit", "criteria", "created_at"], row, strict=True)) if row else None


def record_reply(conn: psycopg.Connection, lead_id: str, note: str = "") -> int:
    return conn.execute(
        "INSERT INTO replies (lead_id, note) VALUES (%s, %s) RETURNING reply_id", [lead_id, note]
    ).fetchone()[0]


def outreach(conn: psycopg.Connection, lead_id: str) -> dict:
    """What we've done with this lead: drafts, what was sent and when, and whether they replied since."""
    drafts = conn.execute(
        f"""
        SELECT d.draft_id, d.subject, d.created_at, s.sent_at FROM drafts d
        LEFT JOIN sends s ON s.draft_id = d.draft_id AND s.status = 'sent' AND s.tenant_id = {TENANT}
        WHERE d.lead_id = %s AND d.tenant_id = {TENANT} ORDER BY d.draft_id
        """,
        [lead_id],
    ).fetchall()
    sent = [(f"D{n}", subject, sent_at) for n, subject, _, sent_at in drafts if sent_at is not None]
    last_sent = max((t for *_, t in sent), default=None)
    last_reply = conn.execute(
        f"SELECT max(received_at) FROM replies WHERE lead_id = %s AND tenant_id = {TENANT}", [lead_id]
    ).fetchone()[0]
    return {
        "drafts_unsent": [f"D{n}" for n, _, _, sent_at in drafts if sent_at is None],
        "sent": [{"draft_id": d, "subject": s, "sent_at": t} for d, s, t in sent],
        "last_contacted_at": last_sent,
        "last_reply_at": last_reply,
        "replied_since_last_contact": bool(last_sent and last_reply and last_reply > last_sent),
    }


def followups_due(conn: psycopg.Connection, days: int) -> list[dict]:
    """Leads we emailed more than `days` days ago with no reply since. A follow-up is itself a send,
    so it resets the clock: no one is chased twice in the same window."""
    rows = conn.execute(
        f"""
        WITH last_send AS (
            SELECT d.lead_id, max(s.sent_at) AS last_sent, count(*) AS emails_sent
            FROM sends s JOIN drafts d ON d.draft_id = s.draft_id
            WHERE s.status = 'sent' AND s.tenant_id = {TENANT} AND d.tenant_id = {TENANT} GROUP BY d.lead_id
        )
        SELECT l.lead_id, l.name, l.company, ls.last_sent, ls.emails_sent
        FROM last_send ls JOIN leads l ON l.lead_id = ls.lead_id AND l.tenant_id = {TENANT}
        WHERE ls.last_sent < now() - make_interval(days => %s)
          AND NOT EXISTS (SELECT 1 FROM replies r WHERE r.lead_id = ls.lead_id AND r.received_at > ls.last_sent
                          AND r.tenant_id = {TENANT})
        ORDER BY ls.last_sent
        """,
        [days],
    ).fetchall()
    cols = ["lead_id", "name", "company", "last_contacted_at", "emails_sent"]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def lock_send(conn: psycopg.Connection, draft_id: str) -> dict:
    """Lock this draft's sends row until the current transaction ends, and return it, freshly read.

    NOWAIT: if another process holds the lock, raise psycopg.errors.LockNotAvailable at once instead
    of waiting. Postgres drops a dead process's locks when its connection closes, so a held lock
    means a LIVE owner, which `status = 'pending'` alone can't tell you.
    """
    row = conn.execute(
        f"SELECT status, sent_at FROM sends WHERE draft_id = %s AND tenant_id = {TENANT} FOR UPDATE NOWAIT",
        [_draft_number(draft_id)],
    ).fetchone()
    return {"status": row[0], "sent_at": row[1]}


# --- ownership (D7 M2): which tenant owns a resource -----------------------------------------

_OWNED = {"lead": ("leads", "lead_id"), "run": ("runs", "run_id"), "draft": ("drafts", "draft_id")}


def owner(conn: psycopg.Connection, kind: str, key: Any) -> str | None:
    """tenant_id of a lead / run / draft IF it belongs to the connection's tenant, else None.

    Filtered like every other query: another tenant's resource looks exactly like a missing one.
    The API still passes the result to can_access(): two independent checks, either one stops a leak."""
    table, column = _OWNED[kind]
    if kind == "draft":
        key = _draft_number(key)
        if key is None:
            return None
    row = conn.execute(f"SELECT tenant_id FROM {table} WHERE {column} = %s AND tenant_id = {TENANT}", [key]).fetchone()
    return row[0] if row else None


# --- limits + usage (D7 M3) -----------------------------------------------------------------
# "Today" = since UTC midnight. All filtered on the connection's tenant like everything else.

TODAY = "date_trunc('day', now(), 'UTC')"


def tenant_limits(conn: psycopg.Connection) -> tuple[int, int]:
    """(max_runs_per_day, daily_token_budget) of the connection's tenant."""
    return conn.execute(
        f"SELECT max_runs_per_day, daily_token_budget FROM tenants WHERE tenant_id = {TENANT}"
    ).fetchone()


def runs_today(conn: psycopg.Connection) -> int:
    return conn.execute(f"SELECT count(*) FROM runs WHERE tenant_id = {TENANT} AND started_at >= {TODAY}").fetchone()[0]


def tokens_today(conn: psycopg.Connection) -> int:
    return conn.execute(
        f"SELECT coalesce(sum(tokens), 0) FROM usage WHERE tenant_id = {TENANT} AND created_at >= {TODAY}"
    ).fetchone()[0]


def record_usage(conn: psycopg.Connection, tokens: int, endpoint: str, run_id: UUID | None = None) -> None:
    conn.execute("INSERT INTO usage (tokens, endpoint, run_id) VALUES (%s, %s, %s)", [tokens, endpoint, run_id])
