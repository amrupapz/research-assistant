"""Agent state (D4 M0 sketch): runs, steps, findings, qualifications, drafts, approvals, sends.

ON DELETE: RESTRICT (the default) everywhere, so history can't silently vanish; CASCADE only
for "part of" (a finding's citations). Nothing points INTO chunks: citations are snapshots.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE runs (
        run_id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        task                  text NOT NULL,
        status                text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'failed')),
        model                 text NOT NULL,
        agent_prompt_version  text NOT NULL,
        answer                text,
        started_at            timestamptz NOT NULL DEFAULT now(),
        finished_at           timestamptz
    );

    -- One row per model turn AND per tool call, in order: enough to rebuild the conversation (M4).
    CREATE TABLE run_steps (
        run_id      uuid NOT NULL REFERENCES runs,
        seq         int  NOT NULL,
        turn        int  NOT NULL,
        kind        text NOT NULL CHECK (kind IN ('model_turn', 'tool_call')),
        call_id     text,
        tool        text,
        payload     jsonb NOT NULL,  -- model_turn: output items (incl. reasoning) | tool_call: the call
        decision    text CHECK (decision IN ('allow', 'deny', 'needs_approval')),
        approved    boolean,
        status      text,
        output      jsonb,           -- tool_call: what the model was sent back
        created_at  timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (run_id, seq),
        CHECK (kind = 'model_turn' OR (call_id IS NOT NULL AND tool IS NOT NULL))
    );

    CREATE TABLE research_findings (
        finding_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        company         text NOT NULL,
        question        text NOT NULL,
        answer          text NOT NULL,
        confidence      text NOT NULL CHECK (confidence IN ('low', 'medium', 'high')),
        needs_sources   boolean NOT NULL,
        prompt_version  text NOT NULL,
        run_id          uuid REFERENCES runs,
        created_at      timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX research_findings_company_idx ON research_findings (lower(company), created_at);  -- M6 reuse

    CREATE TABLE finding_citations (
        finding_id  bigint NOT NULL REFERENCES research_findings ON DELETE CASCADE,
        chunk_id    text NOT NULL,   -- snapshot: no FK, re-ingest rewrites chunks
        source_url  text NOT NULL,
        fetched_at  date NOT NULL,
        PRIMARY KEY (finding_id, chunk_id)
    );

    CREATE TABLE qualifications (
        qualification_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        lead_id           text NOT NULL REFERENCES leads,
        fit               text NOT NULL CHECK (fit IN ('strong', 'possible', 'poor')),
        criteria          jsonb NOT NULL,  -- [{id, met, reason, chunk_ids}] as computed by qualify.py
        icp_version       text NOT NULL,
        run_id            uuid REFERENCES runs,
        created_at        timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE drafts (
        draft_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- shown to the model as "D<id>"
        lead_id     text NOT NULL REFERENCES leads,
        to_email    text NOT NULL,   -- snapshot of the lead's email: what you approve is what gets sent
        subject     text NOT NULL,
        body        text NOT NULL,
        run_id      uuid REFERENCES runs,
        created_at  timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE approvals (
        approval_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        draft_id     bigint NOT NULL REFERENCES drafts,
        decision     text NOT NULL CHECK (decision IN ('approved', 'declined')),
        decided_at   timestamptz NOT NULL DEFAULT now(),
        run_id       uuid REFERENCES runs
    );

    -- The M5 backstop: whatever the code does, the DB refuses a second send row for one draft.
    CREATE TABLE sends (
        send_id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        draft_id             bigint NOT NULL UNIQUE REFERENCES drafts,
        approval_id          bigint NOT NULL REFERENCES approvals,
        idempotency_key      text NOT NULL UNIQUE,
        status               text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
        channel              text NOT NULL CHECK (channel IN ('outbox', 'gmail_mcp', 'smtp')),
        provider_message_id  text,
        created_at           timestamptz NOT NULL DEFAULT now(),
        sent_at              timestamptz,
        CHECK (status <> 'sent' OR sent_at IS NOT NULL)
    );
    """)


def downgrade() -> None:
    # Reverse FK order: a table can't be dropped while another still references it.
    op.execute("""
    DROP TABLE sends;
    DROP TABLE approvals;
    DROP TABLE drafts;
    DROP TABLE qualifications;
    DROP TABLE finding_citations;
    DROP TABLE research_findings;
    DROP TABLE run_steps;
    DROP TABLE runs;
    """)
