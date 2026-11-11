"""Search over the chunks table. M2: keyword. M3: vector. M4: hybrid (RRF merge of both)."""

import argparse
from typing import Literal

import psycopg

from research_assistant import tracing
from research_assistant.db import EmbedFn, to_pgvector
from research_assistant.schemas import Chunk

Result = tuple[Chunk, float]

# websearch_to_tsquery: Google-ish syntax ("quoted phrase", -exclude, or). Plain words are ANDed.
# match="any" rewrites the ANDs to ORs: any word may match (used by hybrid; RRF filters the noise).
# ts_rank_cd ("cover density") also rewards query words appearing close together.
TSQUERY = {
    "all": "websearch_to_tsquery('english', %(q)s)",
    "any": "CAST(replace(websearch_to_tsquery('english', %(q)s)::text, ' & ', ' | ') AS tsquery)",
}
KEYWORD_SQL = """
SELECT chunk_id, source_url, fetched_at, title, text, ts_rank_cd(tsv, query) AS score
FROM chunks, {tsquery} AS query
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


def keyword_search(
    conn: psycopg.Connection, query: str, k: int = 5, match: Literal["all", "any"] = "all"
) -> list[Result]:
    """Top-k chunks matching ALL (or ANY) query words after stemming, best first."""
    sql = KEYWORD_SQL.format(tsquery=TSQUERY[match])  # fixed strings only; the query stays a parameter
    return _to_results(conn.execute(sql, {"q": query, "k": k}).fetchall())


def vector_search(conn: psycopg.Connection, query: str, k: int = 5, embed: EmbedFn | None = None) -> list[Result]:
    """Top-k chunks closest in meaning to the query, best first. Costs one embedding call."""
    if embed is None:
        from research_assistant.llm import embed
    [vector] = embed([query])
    return _to_results(conn.execute(VECTOR_SQL, {"v": to_pgvector(vector), "k": k}).fetchall())


def rrf_merge(result_lists: list[list[Result]], k: int = 60, top_k: int | None = None) -> list[Result]:
    """Fuse several ranked lists into one with Reciprocal Rank Fusion.

    ✍️ YOU WRITE THIS (Project 2 core function).

    Contract:
      - result_lists: each is a ranked list of (Chunk, score), best first. Raw scores are
        on different scales across lists, so IGNORE them; use only positions.
      - A chunk at rank r (1 = best) in a list earns 1 / (k + r) from that list.
        Its fused score is the sum over every list it appears in.
      - The same chunk in different lists is identified by chunk_id.
      - Return (Chunk, fused_score) pairs, highest fused score first.
        Ties: smaller chunk_id first (deterministic output).
      - top_k: keep only the first top_k results (None = keep all).

    Hint: a dict keyed by chunk_id to accumulate scores (and remember each Chunk),
          enumerate(..., start=1), and one sort with a key. ~10 lines.
    Run: uv run pytest tests/test_rrf.py -v
    """
    scores: dict[str, float] = {}                                     # 1
    chunks: dict[str, Chunk] = {}                                     # 2
    for results in result_lists:                                      # 3
        for rank, (chunk, _raw_score) in enumerate(results, start=1):  # 4
            cid = chunk.chunk_id                                      # 5
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)       # 6
            chunks.setdefault(cid, chunk)                             # 7

    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))     # 8
    if top_k is not None:                                             # 9
        ordered = ordered[:top_k]
    return [(chunks[cid], scores[cid]) for cid in ordered]            # 10


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 5,
    candidates: int = 20,
    embed: EmbedFn | None = None,
    rrf_k: int = 60,
) -> list[Result]:
    """Keyword (any-word) + vector, top `candidates` from each, fused with RRF -> top k.

    Traced as a retriever; its child embed observation is the model part, the rest is DB time (M4)."""
    with tracing.observe("hybrid_search", as_type="retriever", input={"query": query, "k": k}) as span:
        keyword = keyword_search(conn, query, candidates, match="any")
        vector = vector_search(conn, query, candidates, embed=embed)
        results = rrf_merge([keyword, vector], k=rrf_k, top_k=k)
        span.update(output=[{"chunk_id": c.chunk_id, "score": round(s, 4)} for c, s in results])
        return results


def main() -> None:
    """Try a query: `uv run python -m research_assistant.retrieval "headcount" --mode vector -k 5`."""
    from research_assistant.db import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "any", "vector", "hybrid"], default="keyword")
    args = parser.parse_args()

    with connect() as conn:
        if args.mode in ("keyword", "any"):
            match = "all" if args.mode == "keyword" else "any"
            tsquery = conn.execute(f"SELECT ({TSQUERY[match]})::text", {"q": args.query}).fetchone()[0]
            print(f"tsquery: {tsquery or '(empty: only stopwords)'}\n")
            results = keyword_search(conn, args.query, args.k, match=match)
        elif args.mode == "vector":
            results = vector_search(conn, args.query, args.k)
        else:
            results = hybrid_search(conn, args.query, args.k)
    if not results:
        print("no matches")
    for chunk, score in results:
        preview = " ".join(chunk.text.split()[:14])
        print(f"{score:.4f}  {chunk.chunk_id:<26} {chunk.fetched_at}  {preview}…")


if __name__ == "__main__":
    main()


def fetch_chunks(conn: psycopg.Connection, chunk_ids: list[str]) -> list[Chunk]:
    """Chunks by id, in the order asked; unknown ids are skipped (citations are snapshots)."""
    if not chunk_ids:
        return []
    rows = conn.execute(
        "SELECT chunk_id, source_url, fetched_at, title, text FROM chunks WHERE chunk_id = ANY(%s)", [chunk_ids]
    ).fetchall()
    by_id = {r[0]: Chunk(chunk_id=r[0], source_url=r[1], fetched_at=r[2], title=r[3], text=r[4]) for r in rows}
    return [by_id[c] for c in chunk_ids if c in by_id]
