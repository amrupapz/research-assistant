# Company Research → Sales Agent (agentic engineering, 1-week sprint)

## Big picture

One system grown across 8 projects in 7 days:
D1 P1 reliable model app → D2 P2 RAG w/ citations → D3 P3 tools + approval →
D4 P4 Postgres state + resume → D5 P5 LangGraph workflow → D6 P6 evals + Langfuse →
D7 P7/P8 FastAPI + Docker + GitHub Actions + README.
CURRENT STAGE: Day 7 — P7/P8 (FastAPI + Docker + GitHub Actions + README). P1–P6 done — see LEARNING.md wrap-ups.
Day 7 milestones: not written yet (plan first). CI runs `evals/gate.py` (see P6).
Carried: `agent approve` should show the draft before deciding (see P5 wrap-up "Open").
M8 (Gmail MCP) dropped: Google's Gmail MCP server looks gated to Workspace preview accounts;
MCP experience comes from the Day 7 server. M9 and the rest: "After the week" under Day 6.
Cut for now (log in LEARNING.md "after the week"): streaming, live web scraping,
automated LinkedIn sending (ToS — manual hand-off instead), Temporal, cloud deploy, multi-tenancy.

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
P4 added: alembic (plain-SQL migrations; SQLAlchemy only for its connection).
Later: langgraph (+ langgraph-checkpoint-postgres), langfuse, fastapi, docker.
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
  evaluate.py   pure eval scoring (retrieval_scores, grade_answer, grade_agent_run)
  tools.py      tool registry: Pydantic args -> strict schemas; validate_call / execute
  permissions.py authorize(ToolCall) -> allow | deny | needs_approval (default-deny)
  agent.py      `agent "task"`: loop, gate (validate -> authorize -> approve -> execute), cli_approve
  flow.py       LangGraph workflow: nodes, route_after_qualify, checkpointer, interrupt approval, guarded/failed
  tracing.py    ONLY file importing langfuse: run_trace/observe/start_generation/eval_trace/fetch_observations
  perf.py       pure cost/latency from trace trees: decompose (self-time split), percentile, summaries
  gate.py       regression_gate(baseline, current, tolerance) -> (ok, reasons); snapshot()
tests/                  mock the client/DB, never hit the real API (*_pg tests: local test DB, rolled back)
evals/questions.jsonl agent_cases.jsonl qualify_cases.jsonl judge_labels.jsonl (every case has a category)
evals/run.py  report.py (cost/latency from traces)  gate.py (vs baseline.json)   data/ (saved pages; linear-customers.md = injection test page)
runs/ (one JSONL record per request)   outbox/ ("sent" .eml files, gitignored)
```

Commands: `uv run python -m research_assistant.db` (load + embed data/),
`uv run ask "question"`, `uv run agent "task"`,
`uv run python -m research_assistant.flow L4 | --resume <run_id>`, `uv run agent approve D20 y|n`,
`uv run python evals/run.py --stage retrieval|answers|qualify|agent|flow|judge [--only ids] [--repeat N] [--judge]`,
`--compare`, `--all` (judges answers); then `evals/report.py` and `evals/gate.py [--save-baseline]` (exit 1 = regression).
Baseline + every gated run: `--all --repeat 3` (same --repeat, or the comparison is meaningless).

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

## Day 7 milestones (Project 7 + 8)

Goal (from the roadmap): make it safe and operable for real users, then package it as a portfolio.
The SIMPLEST architecture that meets reliability/security needs: one FastAPI app + Postgres (pgvector)
in Docker Compose, CI in GitHub Actions. No Kubernetes, no cloud deploy unless time is left.

M0 API surface + threat model (no code): endpoints, who may call what, what an attacker would try
   (another tenant's data, skipping approval, huge requests, burning my OpenAI budget).
   Endpoints: POST /ask, POST /runs (start flow for a lead → run_id), GET /runs/{id},
   POST /drafts/{id}/approve, GET /leads/{id}/status, GET /health.
M1 FastAPI app (ask before adding fastapi + uvicorn): Pydantic request/response models, request id,
   body size limit, errors without stack traces or secrets. Long runs don't block a request:
   POST /runs returns run_id, the flow runs in the background, GET /runs/{id} polls; the approve
   endpoint resumes the paused flow (same interrupt() + send_once path as the CLI).
   /health checks the DB → 503 (not a crash) when Postgres is down.
M2 Auth + authorization + tenant isolation: API keys stored HASHED; tenants + roles (member,
   approver) via migration; leads/runs/drafts belong to a tenant and EVERY query filters on it.
   401 = no/bad key, 404 (not 403) for another tenant's resource, only approvers approve.
   ✍️ I WRITE can_access(user, resource) → bool (tenant match + role). No code given; hint only.
M3 Limits + cost controls: per-key rate limit, max runs per tenant per day, daily token budget per
   tenant (refuse with 429 when exceeded, read usage from my run logs/DB), existing retries/max_steps.
M4 Docker: Dockerfile (uv, non-root user, nothing from .env baked into the image) + docker compose
   (app + pgvector Postgres + migrations on start, data volume). Secrets via environment only.
   `docker compose up` → working API. Break it: stop Postgres → /health 503, requests fail cleanly.
M5 GitHub Actions: on push/PR → ruff + pytest with a pgvector Postgres service (fake model: no
   secrets needed). Separate manual (workflow_dispatch) workflow: evals --all + regression_gate vs
   evals/baseline.json, OPENAI/LANGFUSE keys from GitHub Secrets, report uploaded as an artifact.
   Before the first push: check git history contains no secrets; add .env.example.
M6 Ops: RUNBOOK.md (incident → trace_id → Langfuse → run_steps; resume a stuck/failed run; rotate
   keys; what each alert means). A small check script flags error rate / p95 above threshold from
   run logs (Slack webhook = stretch).
M7 Portfolio (P8): README — what it does, architecture diagram, setup in ≤5 commands, design
   decisions + trade-offs, eval table vs baseline, KNOWN FAILURES listed honestly (30% injection,
   follow-up cap, small categories). Demos: API/model failure (retries → failed → resumable),
   interrupted workflow resumed, unauthorized action blocked (cross-tenant + injected "send now"),
   trace screenshots (no secrets). pyproject description, LICENSE.

Order if time runs short: M0, M1, M2, M4, M5, M7 are the core; M3, M6 can be trimmed; MCP is stretch.

## Day 7 stretch (with P7/P8)

MCP server: expose search_company, get_lead, qualify_lead via the official Python MCP SDK
(stdio). authorize() runs INSIDE the server; send_email is NOT exposed (or returns
needs_approval without sending). Connect it from Claude Desktop / Claude Code and test.
Tests: tool listing, schema matches Pydantic models, denied tool stays denied.

## Done
P1 (Day 1): reliable `ask` CLI — config, run logs, retries, structured output, 31 tests. See LEARNING.md.
P2 (Day 2): RAG w/ citations — chunking, Postgres FTS + pgvector, RRF hybrid, grounded answers
   (prompt v4), evals (retrieval hit@k + answer checks, --repeat), 98 tests. See LEARNING.md.
P3 (Day 3): tools + approval — hand-rolled tool loop, 4 tools, default-deny authorize, y/N gate
   for send_email (outbox/ only), attack tests + poisoned page, agent trajectory evals (agent-v3,
   33/33), ask switched to responses.create (truncation fix), 221 tests. See LEARNING.md.
P4 (Day 4): Postgres state + resume — Alembic (0001–0003), runs/run_steps (step + effect in one
   transaction), qualify_lead (icp.toml), `agent --resume`, send_once (ledger + idempotency key +
   row lock), memory across days (lead_status, list_followups, reuse < 7 days), isolated evals with
   setup/crash_after (agent-v5, 16/16), 361 tests. M8 dropped, M9 carried. See LEARNING.md.
P5 (Day 5): LangGraph workflow — StateGraph with route_after_qualify (one extra research pass), Postgres
   checkpointer (thread_id = run_id, `flow --resume`), interrupt() approval (`agent approve`, runs
   'waiting', 0004), transient vs permanent errors (DB-only RetryPolicy, failed node), loop vs graph
   evals (flow 8/8, --compare), 415 tests. See LEARNING.md.
P6 (Day 6): evals + Langfuse — tracing (one trace per run, trace_id seeded from run_id, 0005), 51
   labelled cases in 7 categories + qualify stage, claim judge (judge-v4, calibrated on my labels:
   dev 16/16, 0 unsupported missed; unsupported fails answers, partial warns), cost/latency report
   (model = 99% of time), baseline + regression gate (tokens not $; A/A test PASS), icp-v2 fix
   (qualify 72 -> 100%, flow p95 24 -> 15s), 503 tests. See LEARNING.md.
