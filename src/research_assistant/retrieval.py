"""Search over the chunks table. M2: keyword (Postgres full-text)."""

import argparse

import psycopg

from research_assistant.schemas import Chunk

# websearch_to_tsquery: Google-ish syntax ("quoted phrase", -exclude, or). Plain words are ANDed.
# ts_rank_cd ("cover density") also rewards query words appearing close together.
KEYWORD_SQL = """
SELECT chunk_id, source_url, fetched_at, title, text, ts_rank_cd(tsv, query) AS score
FROM chunks, websearch_to_tsquery('english', %(q)s) AS query
WHERE tsv @@ query
ORDER BY score DESC, chunk_id
LIMIT %(k)s
"""


def keyword_search(conn: psycopg.Connection, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
    """Top-k chunks matching ALL query words (after stemming), best first."""
    rows = conn.execute(KEYWORD_SQL, {"q": query, "k": k}).fetchall()
    return [
        (Chunk(chunk_id=r[0], source_url=r[1], fetched_at=r[2], title=r[3], text=r[4]), float(r[5]))
        for r in rows
    ]


def main() -> None:
    """Try a query: `uv run python -m research_assistant.retrieval "Lip-Bu Tan" -k 5`."""
    from research_assistant.db import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    with connect() as conn:
        tsquery = conn.execute("SELECT websearch_to_tsquery('english', %s)::text", (args.query,)).fetchone()[0]
        print(f"tsquery: {tsquery or '(empty: only stopwords)'}\n")
        results = keyword_search(conn, args.query, args.k)
    if not results:
        print("no matches")
    for chunk, score in results:
        preview = " ".join(chunk.text.split()[:14])
        print(f"{score:.4f}  {chunk.chunk_id:<26} {chunk.fetched_at}  {preview}…")


if __name__ == "__main__":
    main()
