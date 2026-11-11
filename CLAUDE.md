# Company Research → Sales Agent (agentic engineering, 1-week sprint)

## Big picture

One system grown across 8 projects in 7 days:
D1 P1 reliable model app → D2 P2 RAG w/ citations → D3 P3 tools + approval →
D4 P4 Postgres state + resume → D5 P5 LangGraph workflow → D6 P6 evals + Langfuse →
D7 P7/P8 FastAPI + Docker + GitHub Actions + README.
CURRENT STAGE: Day 6 — Project 6 (evals + Langfuse). P1–P5 done — see LEARNING.md wrap-ups.
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

## Day 6 milestones (Project 6)

Goal (from the roadmap): a REPRODUCIBLE test suite + observability, not a dashboard. Run it before
every change/deploy and after. Measure: qualification correct, citations support claims, right tools
called, unauthorized actions blocked, latency + cost (model time vs my own code/DB time).

M0 Inventory: map my existing evals (questions.jsonl, agent_cases, flow stage) to the roadmap's
   categories: straightforward, insufficient evidence, conflicting info, invalid tool args,
   prompt injection, permission bypass, freshness. List the gaps. One command runs everything:
   `evals/run.py --all` → one summary table.
M1 Langfuse tracing (ask before adding the dependency; Langfuse Cloud free tier, keys in .env;
   self-hosting waits for Docker on Day 7). One trace per agent/flow run; spans for retrieval,
   every LLM call (tokens, cost, latency) and every tool call (decision, status). trace_id stored
   in my run log + runs table. Tests never send traces (disabled/fake).
M2 Grow the dataset to ~30 labelled cases with a `category` field. Must include the carried items:
   Figma "Dev Mode → dev_tooling" stretch, a citation about the WRONG company, the fake "30%"
   discount page, paraphrases, a permission-bypass attempt, a "no follow-up cap" case.
M3 Claim checker (LLM-as-judge): for each answer/draft claim, does the cited chunk actually support
   it? Judge model from JUDGE_MODEL in .env (stronger than MODEL). Calibrate: hand-label ~10 claims,
   report judge agreement %. A judge I haven't checked is not evidence.
M4 Cost + latency report from traces/run logs: per case and per stage, p50/p95, $ per run (price
   table in config), and where time goes: model vs retrieval/DB vs tools.
M5 Baseline + regression gate: commit evals/baseline.json (pass rates per category, cost, p95).
   ✍️ I WRITE regression_gate(baseline, current, tolerance) → (ok, reasons) — pure function, no code
   given; if I ask you to write it, give me a hint instead. Then fix ONE real failure the evals found
   (e.g. source vetting or draft facts vs citations), re-run everything, compare against baseline.
   Day 7 runs this gate in GitHub Actions.

After the week: M9 LinkedIn hand-off; real email via the Gmail API (gmail.compose, drafts only);
async node timeouts; multi-lead fan-out in the flow.

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
