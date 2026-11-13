"""Tenants, API keys, tenant_id on every tenant-owned table (D7 M2).

The CONNECTION carries the tenant: `set_config('app.tenant_id', ...)` (db.set_tenant). Every query
filters on `tenant_id = current_setting('app.tenant_id')`, and that is also the column DEFAULT, so
an INSERT lands in the caller's tenant without passing it around. Fail closed: with no tenant set,
current_setting() raises, so a forgotten set_tenant is an error, not another tenant's rows.

Shared: chunks (public pages = one knowledge base). Tenant-owned: everything a customer did,
INCLUDING research_findings and qualifications: a shared cache would tell tenant B which companies
tenant A is researching. finding_citations hang off a finding (reached only through its finding_id).

Existing rows go to tenant 'default' (the CLI + evals tenant).

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

TENANT_TABLES = [
    "leads", "runs", "run_steps", "drafts", "approvals", "sends", "replies", "research_findings", "qualifications",
]


def upgrade() -> None:
    op.execute("""
    CREATE TABLE tenants (
        tenant_id   text PRIMARY KEY CHECK (tenant_id ~ '^[a-z0-9_-]{1,32}$'),
        created_at  timestamptz NOT NULL DEFAULT now()
    );
    INSERT INTO tenants (tenant_id) VALUES ('default');

    CREATE TABLE api_keys (
        key_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        tenant_id   text NOT NULL REFERENCES tenants,
        role        text NOT NULL CHECK (role IN ('member', 'approver')),
        key_sha256  text NOT NULL UNIQUE,  -- the key itself is never stored
        label       text NOT NULL DEFAULT '',
        created_at  timestamptz NOT NULL DEFAULT now(),
        revoked_at  timestamptz
    );
    """)
    for table in TENANT_TABLES:
        # Backfill existing rows as 'default', then switch the default to the connection's tenant.
        op.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id text NOT NULL DEFAULT 'default' REFERENCES tenants")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id SET DEFAULT current_setting('app.tenant_id')")
        op.execute(f"CREATE INDEX {table}_tenant_idx ON {table} (tenant_id)")
    op.execute("DROP INDEX research_findings_company_idx")
    op.execute("CREATE INDEX research_findings_company_idx ON research_findings (tenant_id, lower(company), created_at)")


def downgrade() -> None:
    op.execute("DROP INDEX research_findings_company_idx")
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN tenant_id")  # drops its index too
    op.execute("CREATE INDEX research_findings_company_idx ON research_findings (lower(company), created_at)")
    op.execute("DROP TABLE api_keys; DROP TABLE tenants")
