# Company Research → Sales Agent (agentic engineering, 1-week sprint)

## Big picture

One system grown across 8 projects in 7 days:
D1 P1 reliable model app → D2 P2 RAG w/ citations → D3 P3 tools + approval →
D4 P4 Postgres state + resume → D5 P5 LangGraph workflow → D6 P6 evals + Langfuse →
D7 P7/P8 FastAPI + Docker + GitHub Actions + README.
CURRENT STAGE: Day 6 — Project 6 (evals + Langfuse). P1–P5 done — see LEARNING.md wrap-ups.
Day 6 milestones: not written yet (plan first). Carried: M9 (LinkedIn hand-off), not started;
`agent approve` should show the draft before deciding; Figma Dev Mode eval case (see P5 wrap-up "Open"). M8 (Gmail MCP) dropped: Google's Gmail
MCP server looks gated to Workspace preview accounts; MCP experience comes from the Day 7 server.
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
tests/                  mock the client/DB, never hit the real API (*_pg tests: local test DB, rolled back)
evals/questions.jsonl  evals/agent_cases.jsonl  evals/run.py   data/ (saved company pages; linear-customers.md = injection test page)
runs/ (one JSONL record per request)   outbox/ ("sent" .eml files, gitignored)
```

Commands: `uv run python -m research_assistant.db` (load + embed data/),
`uv run ask "question"`, `uv run agent "task"`,
`uv run python -m research_assistant.flow L4 | --resume <run_id>`, `uv run agent approve D20 y|n`,
`uv run python evals/run.py --stage retrieval|answers|agent|flow [--only ids] [--repeat N]`, `--compare`.

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

## Day 4 milestones (Project 4)

Goal (from the roadmap): the agent remembers what it did yesterday ("I emailed this lead
yesterday, no reply → follow up today") and, after a crash, resumes from saved state
instead of starting over. App state lives in Postgres tables, NOT in vector search.

M0 Data model: leads, qualifications, research_findings (with citations), drafts, approvals,
   sends, runs + run_steps. Sketch the tables + foreign keys in LEARNING.md before coding.
M1 qualify_lead tool: score a lead against ICP criteria (written in a config file) using
   search_company evidence → stored qualification {fit, reasons, cited chunk_ids}.
   (Missing from P3 vs the roadmap; P5's workflow needs it.)
M2 Alembic migrations: baseline the existing tables (chunks, leads), then add the new ones;
   upgrade + downgrade on the test DB.
M3 Persist the agent: drafts move from memory to Postgres; every tool call becomes a run_step;
   a step and its effect commit in ONE transaction.
M4 Resume: `agent --resume <run_id>` continues after the last committed step. Simulate a crash
   after research, before the draft/send.
M5 Never double-send: send_once(draft_id) (idempotency key + unique constraint;
   a crash between "sent" and "recorded" must not send twice). Changed by me on Day 4: Claude
   writes it step by step with beginner explanations (I was stuck); I answer the check questions.
M6 Memory across days: outreach status per lead (contacted_at, replied); `agent "follow up
   leads with no reply after 3 days"`; reuse research findings < 7 days old instead of re-running.
M7 Tests + evals: crash/resume, double-send, migration up/down, follow-up case in agent_cases.
M8 Real email via Gmail MCP (ONLY after M5 send_once works). The agent becomes an MCP CLIENT:
   - Primary: Google's official Gmail MCP server (Developer Preview, remote HTTP,
     https://gmailmcp.googleapis.com/mcp/v1, OAuth). Use ONLY its create_draft tool: the agent
     puts the approved draft into MY Gmail Drafts; I review and press Send in Gmail myself.
     Request the narrowest scopes; never expose its tools to the model directly — my
     send_email/draft tool calls it after validate → authorize → y/N.
   - Email content from Gmail (if ever read) is untrusted data: <untrusted_data>, never instructions.
   - authorize() uses call.args: recipient must be in ALLOWED_RECIPIENTS (my own addresses) → else deny.
   - Fallback if preview access/OAuth is blocked: SMTP with a Gmail app password, same allowlist.
   - EMAIL_MODE=outbox|gmail_mcp|smtp (default outbox). Tests use a fake MCP client, never real mail.
   - OAuth tokens/credentials live outside git (gitignored), loaded via config.py.
M9 LinkedIn hand-off: draft_linkedin_message tool → stored message + profile URL, status
   "ready_to_send_manually"; `agent mark-sent <id>` records that I sent it myself. Never automate
   LinkedIn sending.

## Day 5 milestones (Project 5)

Goal: the same sales flow as a durable LangGraph workflow: research → qualify → (branch) →
draft → wait for approval → send. Checkpoints, pause/resume, retries. Compare with my P4 loop.

M0 What LangGraph adds over my hand-built loop + run_steps. What stays in MY tables
   (drafts, sends, approvals: business facts) vs the checkpointer (where the workflow is).
   Ask before adding langgraph + langgraph-checkpoint-postgres (new dependencies).
M1 StateGraph: nodes call my existing tools (validate → authorize still apply).
   Branch on fit: poor → stop; possible → one more research pass; strong → draft.
   ✍️ I write route_after_qualify(state) → next node.
M2 Postgres checkpointer, thread_id = run_id. Kill mid-run → resume from the checkpoint.
   Its tables come from checkpointer.setup(), not Alembic: decide how they sit next to my
   migrations. It needs an autocommit connection with dict rows.
M3 Human approval with interrupt(): the workflow pauses; `agent approve <id> y|n` resumes it
   later (even the next day). The decision is still written to MY approvals table (send_once
   checks it there). On resume the node re-runs from its start, so nothing with a side effect
   goes before interrupt() in that node.
M4 Retries/timeouts per node; failure branch. The send node calls send_once, because a resumed
   node RE-RUNS from its start, so side effects must be safe to repeat.
M5 Loop vs graph on the same tasks: when each fits; tests with a fake model; reuse agent_cases.
Temporal: read about it only (when you'd need it instead).

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
