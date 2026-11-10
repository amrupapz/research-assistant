"""Memory across days (D4 M6): replies, and full research answers that can be reused.

- replies: one row per reply I record by hand (`python -m research_assistant.outreach replied L1`).
  "Contacted" is NOT a column anywhere: it's derived from sends (latest sent_at per lead), so it
  can't drift from what was actually sent.
- research_findings.assumptions: needed to hand back a stored answer exactly as search_company
  returned it (ResearchAnswer has assumptions; 0002 forgot them).

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE replies (
        reply_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        lead_id      text NOT NULL REFERENCES leads,
        received_at  timestamptz NOT NULL DEFAULT now(),
        note         text NOT NULL DEFAULT ''
    );
    CREATE INDEX replies_lead_idx ON replies (lead_id, received_at);

    ALTER TABLE research_findings ADD COLUMN assumptions jsonb NOT NULL DEFAULT '[]';
    """)


def downgrade() -> None:
    op.execute("""
    ALTER TABLE research_findings DROP COLUMN assumptions;
    DROP TABLE replies;
    """)
