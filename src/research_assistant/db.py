"""Postgres connection + schema. Functions don't commit; the caller owns the transaction."""

import psycopg

from research_assistant.schemas import Chunk

# tsv is computed by Postgres from title + text on every insert/update (stemmed, stopwords dropped).
SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    text PRIMARY KEY,
    source_url  text NOT NULL,
    fetched_at  date NOT NULL,
    title       text NOT NULL,
    text        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""

UPSERT = """
INSERT INTO chunks (chunk_id, source_url, fetched_at, title, text)
VALUES (%(chunk_id)s, %(source_url)s, %(fetched_at)s, %(title)s, %(text)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    source_url = EXCLUDED.source_url,
    fetched_at = EXCLUDED.fetched_at,
    title      = EXCLUDED.title,
    text       = EXCLUDED.text
"""


def connect(url: str | None = None) -> psycopg.Connection:
    if url is None:
        from research_assistant.config import get_settings

        url = get_settings().database_url
    return psycopg.connect(url)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def sync_chunks(conn: psycopg.Connection, chunks: list[Chunk]) -> int:
    """Make the table match `chunks`: upsert each one, delete ids that no longer exist.

    Returns how many stale chunks were deleted. Running it twice changes nothing.
    """
    with conn.cursor() as cur:
        cur.executemany(UPSERT, [c.model_dump() for c in chunks])
        cur.execute("DELETE FROM chunks WHERE chunk_id <> ALL(%s)", ([c.chunk_id for c in chunks],))
        return cur.rowcount


def main() -> None:
    """Load data/ into Postgres: `uv run python -m research_assistant.db`."""
    from research_assistant.ingest import load_chunks

    chunks = load_chunks()
    with connect() as conn:  # commits on clean exit, rolls back on error
        init_schema(conn)
        deleted = sync_chunks(conn, chunks)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"synced {len(chunks)} chunks, deleted {deleted} stale; table now has {total}")


if __name__ == "__main__":
    main()
