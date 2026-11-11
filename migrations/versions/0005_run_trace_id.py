"""runs.trace_id (D6 M1): the Langfuse trace for this run.

Deterministic from run_id (create_trace_id(seed=run_id)), so a resumed or approved run joins the
same trace. NULL = the run was not traced (tracing off), so there is nothing to look up.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN trace_id text")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN trace_id")
