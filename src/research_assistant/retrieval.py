"""Search over the chunks table. M2: keyword (Postgres full-text). M3: vector (pgvector)."""

import argparse

import psycopg

from research_assistant.db import EmbedFn, to_pgvector
from research_assistant.schemas import Chunk

Result = tuple[Chunk, float]

# websearch_to_tsquery: Google-ish syntax ("quoted phrase", -exclude, or). Plain words are ANDed.
# ts_rank_cd ("cover density") also rewards query words appearing close together.
KEYWORD_SQL = """
SELECT chunk_id, source_url, fetched_at, title, text, ts_rank_cd(tsv, query) AS score
FROM chunks, websearch_to_tsquery('english', %(q)s) AS query
WHERE tsv @@ query
ORDER BY score DESC, chunk_id
LIMIT %(k)s
"""

# <=> is cosine distance (0 = same direction, 2 = opposite). Score = 1 - distance = cosine similarity.
VECTOR_SQL = """
SELECT chunk_id, source_url, fetched_at, title, text, 1 - (embedding <=> %(v)s::vector) AS score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %(v)s::vector, chunk_id
LIMIT %(k)s
"""


def _to_results(rows) -> list[Result]:
    return [
        (Chunk(chunk_id=r[0], source_url=r[1], fetched_at=r[2], title=r[3], text=r[4]), float(r[5]))
        for r in rows
    ]


def keyword_search(conn: psycopg.Connection, query: str, k: int = 5) -> list[Result]:
    """Top-k chunks matching ALL query words (after stemming), best first."""
    return _to_results(conn.execute(KEYWORD_SQL, {"q": query, "k": k}).fetchall())


def vector_search(conn: psycopg.Connection, query: str, k: int = 5, embed: EmbedFn | None = None) -> list[Result]:
    """Top-k chunks closest in meaning to the query, best first. Costs one embedding call."""
    if embed is None:
        from research_assistant.llm import embed
    [vector] = embed([query])
    return _to_results(conn.execute(VECTOR_SQL, {"v": to_pgvector(vector), "k": k}).fetchall())


def main() -> None:
    """Try a query: `uv run python -m research_assistant.retrieval "headcount" --mode vector -k 5`."""
    from research_assistant.db import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "vector"], default="keyword")
    args = parser.parse_args()

    with connect() as conn:
        if args.mode == "keyword":
            tsquery = conn.execute("SELECT websearch_to_tsquery('english', %s)::text", (args.query,)).fetchone()[0]
            print(f"tsquery: {tsquery or '(empty: only stopwords)'}\n")
            results = keyword_search(conn, args.query, args.k)
        else:
            results = vector_search(conn, args.query, args.k)
    if not results:
        print("no matches")
    for chunk, score in results:
        preview = " ".join(chunk.text.split()[:14])
        print(f"{score:.4f}  {chunk.chunk_id:<26} {chunk.fetched_at}  {preview}…")


if __name__ == "__main__":
    main()
