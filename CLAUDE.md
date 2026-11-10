# Company Research → Sales Agent (agentic engineering, 1-week sprint)

## Big picture

One system grown across 8 projects in 7 days:
D1 P1 reliable model app → D2 P2 RAG w/ citations → D3 P3 tools + approval →
D4 P4 Postgres state + resume → D5 P5 LangGraph workflow → D6 P6 evals + Langfuse →
D7 P7/P8 FastAPI + Docker + GitHub Actions + README.
CURRENT STAGE: Day 3 — Project 3 (tools + approval). P1, P2 done — see LEARNING.md wrap-ups.
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
  evaluate.py   pure eval scoring (retrieval_scores, grade_answer)
tests/                  mock the client/DB, never hit the real API (*_pg tests: local test DB, rolled back)
evals/questions.jsonl   evals/run.py   data/ (saved company pages)   runs/ (one JSONL record per request)
```

Commands: `uv run python -m research_assistant.db` (load + embed data/),
`uv run ask "question"`, `uv run python evals/run.py --stage retrieval|answers [--only ids] [--repeat N]`.

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
- Strict grounding: factual claims must cite chunk_ids (checked in code). Missing or conflicting
  evidence → say so, needs_sources=true.
- Evaluate retrieval and answers SEPARATELY. Bump PROMPT_VERSION when prompt/schema changes.

## Day 3 milestones (Project 3)

M0 Tool calling by hand: one tool schema → see the function_call item → send back function_call_output
M1 3 tools: search_company (wraps grounded ask), get_lead (small leads table), draft_email; Pydantic arg models
M2 Agent loop: model proposes → validate args → check permission → execute → feed result back; max_steps; log every tool call
M3 Approval gate: send_email needs my y/n in the CLI; "sending" writes to outbox/, never real email
M4 Attack it: bad args, unknown tool, unauthorized tool, prompt injection hidden in a data/ page
M5 Tests + evals: expected tool + args per case; blocked actions stay blocked
✍️ I WRITE authorize(tool_call) → allow | deny | needs_approval (no code given; I type it)
Open items from P2 are in LEARNING.md.

## Done
P1 (Day 1): reliable `ask` CLI — config, run logs, retries, structured output, 31 tests. See LEARNING.md.
P2 (Day 2): RAG w/ citations — chunking, Postgres FTS + pgvector, RRF hybrid, grounded answers
   (prompt v4), evals (retrieval hit@k + answer checks, --repeat), 98 tests. See LEARNING.md.
