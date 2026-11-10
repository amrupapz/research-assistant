"""Postgres connection + schema. Functions don't commit; the caller owns the transaction."""

from collections.abc import Callable

import psycopg

from research_assistant.schemas import Chunk, Lead

EmbedFn = Callable[[list[str]], list[list[float]]]

# tsv is computed by Postgres from title + text on every insert/update (stemmed, stopwords dropped).
# embedding is filled later by embed_missing(); NULL means "needs embedding".
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    text PRIMARY KEY,
    source_url  text NOT NULL,
    fetched_at  date NOT NULL,
    title       text NOT NULL,
    text        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector({dims});
CREATE TABLE IF NOT EXISTS leads (
    lead_id  text PRIMARY KEY,
    name     text NOT NULL,
    title    text NOT NULL,
    company  text NOT NULL,
    email    text NOT NULL
);
"""

# Fictional contacts, one per company in data/. example.com is reserved: mail there goes nowhere.
SEED_LEADS = [
    Lead(lead_id="L1", name="Dana Kim", title="VP Engineering", company="Figma", email="dana.kim@example.com"),
    Lead(lead_id="L2", name="Priya Raman", title="Head of Platform", company="Intel", email="priya.raman@example.com"),
    Lead(lead_id="L3", name="Marco Silva", title="CTO", company="Linear", email="marco.silva@example.com"),
    Lead(lead_id="L4", name="Jordan Lee", title="Engineering Manager", company="Vercel", email="jordan.lee@example.com"),
]

# On conflict, `chunks.x` is the existing row and `EXCLUDED.x` the new one.
# Changed content -> stale embedding -> reset to NULL so embed_missing() redoes it.
UPSERT = """
INSERT INTO chunks (chunk_id, source_url, fetched_at, title, text)
VALUES (%(chunk_id)s, %(source_url)s, %(fetched_at)s, %(title)s, %(text)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    source_url = EXCLUDED.source_url,
    fetched_at = EXCLUDED.fetched_at,
    title      = EXCLUDED.title,
    text       = EXCLUDED.text,
    embedding  = CASE
        WHEN (chunks.title, chunks.text) IS DISTINCT FROM (EXCLUDED.title, EXCLUDED.text) THEN NULL
        ELSE chunks.embedding
    END
"""


def connect(url: str | None = None) -> psycopg.Connection:
    if url is None:
        from research_assistant.config import get_settings

        url = get_settings().database_url
    return psycopg.connect(url)


def init_schema(conn: psycopg.Connection, dims: int) -> None:
    conn.execute(SCHEMA.format(dims=int(dims)))


def sync_chunks(conn: psycopg.Connection, chunks: list[Chunk]) -> int:
    """Make the table match `chunks`: upsert each one, delete ids that no longer exist.

    Returns how many stale chunks were deleted. Running it twice changes nothing.
    """
    with conn.cursor() as cur:
        cur.executemany(UPSERT, [c.model_dump() for c in chunks])
        cur.execute("DELETE FROM chunks WHERE chunk_id <> ALL(%s)", ([c.chunk_id for c in chunks],))
        return cur.rowcount


def seed_leads(conn: psycopg.Connection, leads: list[Lead] = SEED_LEADS) -> None:
    """Upsert the seed leads (running it twice changes nothing)."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO leads (lead_id, name, title, company, email)
            VALUES (%(lead_id)s, %(name)s, %(title)s, %(company)s, %(email)s)
            ON CONFLICT (lead_id) DO UPDATE SET
                name = EXCLUDED.name, title = EXCLUDED.title,
                company = EXCLUDED.company, email = EXCLUDED.email
            """,
            [lead.model_dump() for lead in leads],
        )


def fetch_lead(conn: psycopg.Connection, lead_id: str) -> Lead | None:
    row = conn.execute(
        "SELECT lead_id, name, title, company, email FROM leads WHERE lead_id = %s", (lead_id,)
    ).fetchone()
    return None if row is None else Lead(**dict(zip(Lead.model_fields, row, strict=True)))


def embedding_input(title: str, text: str) -> str:
    """What we embed for a chunk: the title gives context the chunk text may lack (M1 gotcha)."""
    return f"{title}\n\n{text}"


def to_pgvector(vector: list[float]) -> str:
    """pgvector's text format '[0.1,0.2,...]'; cast with ::vector in SQL. Avoids an extra dependency."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def embed_missing(conn: psycopg.Connection, embed: EmbedFn) -> int:
    """Embed only chunks whose embedding is NULL (new or changed). Returns how many were embedded."""
    rows = conn.execute("SELECT chunk_id, title, text FROM chunks WHERE embedding IS NULL ORDER BY chunk_id").fetchall()
    if not rows:
        return 0
    vectors = embed([embedding_input(title, text) for _, title, text in rows])
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector WHERE chunk_id = %s",
            [(to_pgvector(v), chunk_id) for (chunk_id, _, _), v in zip(rows, vectors, strict=True)],
        )
    return len(rows)


def main() -> None:
    """Load data/ into Postgres and embed new/changed chunks: `uv run python -m research_assistant.db`."""
    from research_assistant.config import get_settings
    from research_assistant.ingest import load_chunks
    from research_assistant.llm import embed

    chunks = load_chunks()
    with connect() as conn:  # commits on clean exit, rolls back on error
        init_schema(conn, get_settings().embedding_dims)
        deleted = sync_chunks(conn, chunks)
        embedded = embed_missing(conn, embed)
        seed_leads(conn)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"synced {len(chunks)} chunks, deleted {deleted} stale, embedded {embedded}; table now has {total}")
    print(f"seeded {len(SEED_LEADS)} leads")


if __name__ == "__main__":
    main()
