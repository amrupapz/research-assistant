"""Agent state in Postgres (D4 M3): runs, steps, drafts, approvals, qualifications.

Plain SQL, no commits: the CALLER owns the transaction. agent.py wraps each step and its effect
in one `with conn.transaction():` so they commit together or not at all.
"""

import json
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from research_assistant.schemas import Draft, Lead, Qualification


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=lambda v: json.dumps(v, default=str))  # dates etc. -> strings


# --- runs + steps ---------------------------------------------------------------------------


def start_run(conn: psycopg.Connection, task: str, model: str, agent_prompt_version: str) -> UUID:
    return conn.execute(
        "INSERT INTO runs (task, model, agent_prompt_version) VALUES (%s, %s, %s) RETURNING run_id",
        [task, model, agent_prompt_version],
    ).fetchone()[0]


def get_run(conn: psycopg.Connection, run_id: UUID) -> dict | None:
    row = conn.execute(
        "SELECT task, status, model, agent_prompt_version FROM runs WHERE run_id = %s", [run_id]
    ).fetchone()
    return dict(zip(["task", "status", "model", "agent_prompt_version"], row, strict=True)) if row else None


def mark_running(conn: psycopg.Connection, run_id: UUID) -> None:
    conn.execute("UPDATE runs SET status = 'running', finished_at = NULL WHERE run_id = %s", [run_id])


def finish_run(conn: psycopg.Connection, run_id: UUID, status: str, answer: str | None = None) -> None:
    conn.execute(
        "UPDATE runs SET status = %s, answer = %s, finished_at = now() WHERE run_id = %s", [status, answer, run_id]
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
        "FROM run_steps WHERE run_id = %s ORDER BY seq",
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
    row = conn.execute("SELECT draft_id, lead_id, to_email, subject, body FROM drafts WHERE draft_id = %s", [n]).fetchone()
    return _draft(row) if row else None


def list_drafts(conn: psycopg.Connection, run_id: UUID | None = None) -> list[Draft]:
    """All drafts (or one run's), oldest first."""
    sql = "SELECT draft_id, lead_id, to_email, subject, body FROM drafts"
    rows = conn.execute(sql + " WHERE run_id = %s ORDER BY draft_id", [run_id]).fetchall() if run_id else (
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
        "SELECT draft_id, %s, %s FROM drafts WHERE draft_id = %s RETURNING approval_id",  # 0 rows if no draft
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
        "SELECT fit, criteria, icp_version, created_at FROM qualifications WHERE lead_id = %s "
        "ORDER BY created_at DESC, qualification_id DESC LIMIT 1",
        [lead_id],
    ).fetchone()
    return dict(zip(["fit", "criteria", "icp_version", "created_at"], row, strict=True)) if row else None
