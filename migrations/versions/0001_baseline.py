"""Baseline: the tables that existed before Alembic (P2 chunks, P3 leads).

An existing database already has these: mark it with `uv run alembic stamp 0001` instead of
running this. A fresh database (tests, Docker later) runs it like any other migration.

Revision ID: 0001
Revises:
"""

from alembic import context, op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _dims() -> int:
    dims = context.config.attributes.get("embedding_dims")
    if dims is None:
        from research_assistant.config import get_settings

        dims = get_settings().embedding_dims
    return int(dims)


def upgrade() -> None:
    # tsv is computed by Postgres from title + text (stemmed, stopwords dropped).
    # embedding NULL means "needs embedding" (db.embed_missing).
    op.execute(f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE chunks (
        chunk_id    text PRIMARY KEY,
        source_url  text NOT NULL,
        fetched_at  date NOT NULL,
        title       text NOT NULL,
        text        text NOT NULL,
        tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || text)) STORED,
        embedding   vector({_dims()})
    );
    CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
    CREATE TABLE leads (
        lead_id  text PRIMARY KEY,
        name     text NOT NULL,
        title    text NOT NULL,
        company  text NOT NULL,
        email    text NOT NULL
    );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE leads; DROP TABLE chunks;")  # extension left in place: other DBs/tools may use it
