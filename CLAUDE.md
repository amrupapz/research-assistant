# Company Research → Sales Agent (agentic engineering, 1-week sprint)

## Big picture

One system grown across 8 projects in 7 days:
D1 P1 reliable model app → D2 P2 RAG w/ citations → D3 P3 tools + approval →
D4 P4 Postgres state + resume → D5 P5 LangGraph workflow → D6 P6 evals + Langfuse →
D7 P7/P8 FastAPI + Docker + GitHub Actions + README.
CURRENT STAGE: Day 2 — Project 2 (RAG with citations). P1 done — see LEARNING.md wrap-up.
Cut for now (log in LEARNING.md "after the week"): streaming, live web scraping,
real email/LinkedIn, MCP, Temporal, cloud deploy, multi-tenancy.

## How to work with me (learning on a deadline)

- Plan before code; wait for my OK. Keep plans short.
- Before each milestone: explain the concept in ≤5 sentences + what breaks without it.
- Each project: leave ONE core function for me to write (stub + failing test + hint).
- Simplest thing that works. No new dependency without asking.
- After changes: what changed and why, then 2 check questions.
- Append to LEARNING.md: concept, my answer, gotchas.
- Before "done": `uv run ruff check . && uv run pytest` passes; then suggest a commit message.

## Stack

Python 3.12, uv, ruff, pytest, openai (Responses API), pydantic, pydantic-settings.
Model name comes from MODEL in .env (currently gpt-6-luna).
P2 adds: Postgres 17 + pgvector (Homebrew, local), psycopg. Embedding model from EMBEDDING_MODEL in .env.
Later: alembic, langgraph, langfuse, fastapi, docker.
Never print or commit the API key.

## Layout

```
src/research_assistant/
  config.py     settings from .env
  llm.py        ONLY file that calls OpenAI (responses + embeddings)
  prompts.py    system prompt + PROMPT_VERSION
  schemas.py    Pydantic models
  log.py        structured JSON logging
  cli.py        `ask "question"`
  ingest.py     data/ pages → chunks (chunk_id, source_url, fetched_at)
  retrieval.py  keyword search, vector search, hybrid merge
  db.py         Postgres connection + schema
tests/                  mock the client/DB, never hit the real API
evals/questions.jsonl   data/ (saved company pages)   runs/ (one JSONL record per request)
```

## Rules for every model call

- Responses API via llm.py. Client: max_retries=0, explicit timeout — I build retries myself.
- Retry with backoff + jitter only on APITimeoutError, RateLimitError, 5xx. Never 400/401.
- Log: request_id, model, PROMPT_VERSION, latency_ms, input/output/cached tokens, status, retries.
- status == "incomplete" (max_output_tokens) = truncated → error, not an answer.
- System prompt static (cacheable); per-request data goes in the user message.
- Never read, open, or print .env. Settings come from config.py only.
- gpt-6-luna is a reasoning model: `temperature` → 400. Don't send it.
- Embedding calls also go through llm.py (e.g. `embed(texts)`) and get a run-log line too.

## RAG rules

- Every chunk carries chunk_id, source_url, fetched_at. No chunk without a source.
- Retrieved text goes in the user message inside <untrusted_data> tags (data, not instructions;
  also keeps the system prompt static/cacheable).
- Factual claims must cite chunk_ids. Missing or conflicting evidence → say so, needs_sources=true.
- Evaluate retrieval and answers SEPARATELY. Bump PROMPT_VERSION when prompt/schema changes.

## Day 2 milestones (Project 2)

M0 Save pages into data/ that answer evals q06–q10 (funding, headcount, CEO, pricing, launch),
   each with a url + fetched date header. Include one outdated + one conflicting page on purpose.
M1 Ingest + chunking with overlap; inspect chunk sizes
M2 Keyword search (Postgres full-text): try exact names/numbers
M3 pgvector: embed chunks via llm.embed, vector search: try paraphrased questions
M4 Hybrid search (Reciprocal Rank Fusion) + top-k   ✍️ I WRITE the RRF merge myself, no code given
M5 Grounded answer: ResearchAnswer gains citations [{chunk_id, source_url}]; abstain on thin evidence
M6 Evals: retrieval hit rate@k on q06–q10, then answer + citation checks; fake-DB/client tests

## Done
P1 (Day 1): reliable `ask` CLI — config, run logs, retries, structured output, 31 tests. See LEARNING.md.
