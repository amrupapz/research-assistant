"""Per-tenant daily limits + a token usage ledger (D7 M3).

tenants.max_runs_per_day / daily_token_budget: each customer's plan (defaults for existing tenants).
usage: one row per metered API request or background job (tokens = input + output of every model
and embedding call inside it). The budget check sums today's rows BEFORE the next model call.
The run logs can't do this: their lines don't carry the tenant.

Revision ID: 0007
Revises: 0006
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    ALTER TABLE tenants
        ADD COLUMN max_runs_per_day    integer NOT NULL DEFAULT 50 CHECK (max_runs_per_day >= 0),
        ADD COLUMN daily_token_budget  bigint  NOT NULL DEFAULT 1000000 CHECK (daily_token_budget >= 0);

    CREATE TABLE usage (
        usage_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        tenant_id   text NOT NULL DEFAULT current_setting('app.tenant_id') REFERENCES tenants,
        run_id      uuid REFERENCES runs,
        endpoint    text NOT NULL,
        tokens      bigint NOT NULL CHECK (tokens >= 0),
        created_at  timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX usage_tenant_day_idx ON usage (tenant_id, created_at);
    CREATE INDEX runs_tenant_started_idx ON runs (tenant_id, started_at);
    """)


def downgrade() -> None:
    op.execute("""
    DROP INDEX runs_tenant_started_idx;
    DROP TABLE usage;
    ALTER TABLE tenants DROP COLUMN max_runs_per_day, DROP COLUMN daily_token_budget;
    """)
