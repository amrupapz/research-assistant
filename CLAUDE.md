# Company Research → Sales Agent (agentic engineering, 1-week sprint)

## Big picture

One system grown across 8 projects in 7 days:
D1 P1 reliable model app → D2 P2 RAG w/ citations → D3 P3 tools + approval →
D4 P4 Postgres state + resume → D5 P5 LangGraph workflow → D6 P6 evals + Langfuse →
D7 P7/P8 FastAPI + Docker + GitHub Actions + README.
CURRENT STAGE: Day 1 — Project 1.
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
Later: Postgres + pgvector, alembic, langgraph, langfuse, fastapi, docker.
Never print or commit the API key.

## Layout

src/research_assistant/
config.py settings from .env
llm.py ONLY file that calls OpenAI
prompts.py system prompt + PROMPT_VERSION
schemas.py Pydantic models
log.py structured JSON logging
cli.py `ask "question"`
tests/ (mock the client, never hit the real API)
evals/questions.jsonl data/ (saved company pages) runs/ (one JSONL record per request)

## Rules for every model call

- Responses API via llm.py. Client: max_retries=0, explicit timeout — I build retries myself.
- Retry with backoff + jitter only on APITimeoutError, RateLimitError, 5xx. Never 400/401.
- Log: request_id, model, PROMPT_VERSION, latency_ms, input/output/cached tokens, status, retries.
- status == "incomplete" (max_output_tokens) = truncated → error, not an answer.
- System prompt static (cacheable); per-request data goes in the user message.

## Day 1 milestones (Project 1)

M0 One raw HTTP call with httpx: see request/response JSON, 401/400/429 (30 min)
M1 SDK call + system prompt + config + CLI (company research domain)
M2 Structured logs + runs/\*.jsonl
M3 Timeout, retries, rate limit, truncation handling
M4 Structured output: responses.parse with Pydantic schema
{answer, confidence, assumptions, needs_sources}
M5 Same question ×5 + 2 temperatures → compare; seed evals/questions.jsonl with
10 company questions incl. ones the model can't know (recent funding, headcount)
M6 3–4 unit tests (retry, truncation, schema) with a mocked client
