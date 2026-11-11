"""A run can wait for a human (D5 M3): runs.status gains 'waiting'.

A LangGraph flow paused at interrupt() (approval) is neither done nor failed: it's 'waiting' until
`agent approve <draft_id> y|n` resumes it, maybe the next day.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    ALTER TABLE runs DROP CONSTRAINT runs_status_check;
    ALTER TABLE runs ADD CONSTRAINT runs_status_check
        CHECK (status IN ('running', 'waiting', 'done', 'failed'));
    """)


def downgrade() -> None:
    # The old CHECK can't hold 'waiting': those runs become 'failed' (their checkpoint still exists).
    op.execute("""
    UPDATE runs SET status = 'failed' WHERE status = 'waiting';
    ALTER TABLE runs DROP CONSTRAINT runs_status_check;
    ALTER TABLE runs ADD CONSTRAINT runs_status_check CHECK (status IN ('running', 'done', 'failed'));
    """)
