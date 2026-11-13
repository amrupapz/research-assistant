# Research Assistant → Sales Agent

An AI agent that researches a company from saved web pages, qualifies the lead against an ideal
customer profile, and drafts an outreach email. A human approves every send, and each email is sent
**exactly once**, even if a process crashes.

Built in 7 days as one system grown in stages: a reliable model client, RAG with citations, tools
with a permission gate, durable Postgres state, a LangGraph workflow, evals with tracing, and finally
an authenticated multi-tenant API in Docker with CI. The day-by-day notes, including what went wrong,
are in [LEARNING.md](LEARNING.md).

**What it is not:** it doesn't scrape the live web, it doesn't send real email (a "send" writes an
`.eml` file to `outbox/`), and it isn't deployed anywhere.

## How it works

```mermaid
flowchart LR
    client([API client]) -->|Bearer key| api[FastAPI<br/>auth · limits · tenant · 404/403]
    api -->|POST /runs: 202 + run_id| bg[background task]
    bg --> flow

    subgraph flow [LangGraph workflow, checkpointed in Postgres]
        direction LR
        load[load_lead] --> research[research<br/>RAG + citations]
        research --> qualify[qualify<br/>vs icp.toml]
        qualify -->|possible: 1 extra pass| research
        qualify -->|poor| stop([stop])
        qualify -->|strong| draft[draft email]
        draft --> approve{{approve<br/>interrupt: waits for a human}}
        approve -->|yes| send[send_once<br/>ledger + idempotency key]
        approve -->|no| done([declined])
    end

    client -->|POST /drafts/id/approve| api
    api -->|resume| approve

    research <--> pg[(Postgres + pgvector<br/>chunks · state · checkpoints)]
    research --> llm[llm.py<br/>the only OpenAI caller]
    llm -.-> lf[Langfuse traces]
```

- **Research** runs hybrid search (Postgres full-text + pgvector, merged with reciprocal rank fusion)
  over saved pages in `data/`. The model must cite a `chunk_id` for every factual claim. Code checks
  the citations, and an answer without enough evidence says so (`needs_sources: true`).
- **Qualify** scores the lead against `icp.toml`: the title is checked in code, each criterion is
  judged by the model against retrieved evidence, and the final fit is computed in code.
- **Tools** go through one gate: validate the arguments (Pydantic strict schemas) → `authorize()`
  (default-deny) → human approval for `send_email` → execute.
- **Approval** uses LangGraph's `interrupt()`. The run pauses as `waiting` and can be approved from
  another process the next day. `GET /runs/{id}` shows the draft *before* you approve it.
- **Sending** claims a ledger row with a unique idempotency key and a row lock before writing the
  email, so retries, double clicks and crash-and-resume can't send twice.

## Quickstart (Docker)

Needs Docker and an OpenAI API key.

```bash
git clone https://github.com/amrupapz/research-assistant.git && cd research-assistant
cp .env.example .env
docker compose up --build -d
docker compose run --rm app python -m research_assistant.db
KEY=$(docker compose exec -T app keys create-key --tenant default --role approver --label me | tail -1)
```

What each line does:
1. Clone the repo.
2. Create `.env`. Set `OPENAI_API_KEY` and `POSTGRES_PASSWORD` in it before going on.
3. Start Postgres (pgvector), run the migrations, and start the API on `127.0.0.1:8000`.
4. Load and embed `data/`. This makes real embedding calls and costs a few cents.
5. Create an API key and keep it in `$KEY`. It isn't printed; `echo $KEY` shows it if you need to
   save it.

Then start a run for lead L1:

```bash
curl -s -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -X POST localhost:8000/runs -d '{"lead_id": "L1"}'
```

That returns `202 {"run_id": ...}`. Poll `GET /runs/{run_id}` until the status is `waiting`. The
response then includes the draft, which you read before deciding:

```bash
curl -s -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -X POST localhost:8000/drafts/D1/approve -d '{"approved": false}'
```

Interactive docs are at `http://localhost:8000/docs`. For local development without Docker, use
`uv sync` with a local Postgres 17 + pgvector, then `uv run pytest`.

## API

| Endpoint | Who | What |
|---|---|---|
| `GET /health` | anyone | DB ping: 200, or 503 when Postgres is down (never a crash) |
| `POST /ask` | member | a grounded answer with citations and their sources |
| `POST /runs` | member | start the workflow for a lead → `202 {run_id}`; it runs in the background |
| `GET /runs/{id}` | member | status (`running` / `waiting` / `done` / `failed`), fit, warnings, the draft while waiting |
| `POST /drafts/{id}/approve` | **approver** | approve or decline; resumes the paused run |
| `GET /leads/{id}/status` | member | outreach history, replies, latest qualification and research |

**Status codes:**
- `401`: no key, a bad key, or a revoked key (same message for all three).
- `404`: missing **or another tenant's**. The body is identical in both cases, so a caller can't
  probe for ids.
- `403`: your tenant, but your role can't do it (a member approving).
- `409`: the draft isn't waiting.
- `429`: a limit was reached, with `Retry-After` saying when to try again. Three limits are checked
  before any model call:
  - per-key rate limit: 60 requests a minute
  - runs per tenant per UTC day: 50 by default
  - tokens per tenant per UTC day: 1,000,000 by default
- `413` / `422`: body too large / invalid. The error body never echoes your input.
- Every error is `{error, request_id}`. Stack traces go to the server log, never to the client.

API keys are managed from a CLI only (`keys add-tenant | create-key | list-keys | revoke-key`); there
is no admin endpoint. Keys are stored as SHA-256 hashes. Each tenant's plan is set when it's created:
`keys add-tenant acme --max-runs 20 --daily-tokens 200000`.

## Design decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| My own retry loop (backoff + jitter, only timeouts / 429 / 5xx), SDK retries off | Every attempt is logged and counted; 400/401 are never retried | More code than `max_retries=2` |
| Hybrid search: full-text + vectors, merged with RRF | Exact names (full-text) and paraphrases (vectors) both find the right chunk | Two queries per search |
| Citations checked in code, not only prompted | A claim without a valid `chunk_id` is an error, not an answer | Some correct answers are rejected |
| Default-deny `authorize()`; `send_email` always needs a human | Prompt injection can't add a capability the code doesn't grant | Every send needs a click |
| `send_once`: ledger row + idempotency key + row lock | Exactly-once under retries, double approval and crash-resume | A pending row must be reconciled after a crash |
| LangGraph + Postgres checkpointer, `interrupt()` for approval | Resume after a crash; a human can answer the next day | Two sources of truth: business facts in my tables, position in the checkpoint |
| Tenant on the **connection** (`set_config`), every query filters on it | One mechanism for every table; with no tenant set, queries **error** instead of returning all rows | Must be reset per connection if a pool is ever added |
| Postgres row-level security **not** used | My local role is a superuser, which bypasses RLS, so tests couldn't prove it | The filters are explicit SQL, reviewed by tests |
| Limits checked at the **start** of a request; tokens metered in `llm.py` (a context variable) into a `usage` table | A 429 arrives before any money is spent; usage is per tenant (run logs don't carry the tenant) | A run that starts under budget may finish over it (at most one run, ~6k tokens) |
| In-memory per-key rate limiter (sliding window) | No new infrastructure | Resets on restart and isn't shared between processes; several app containers would need Postgres or Redis |
| Regression gate on **tokens**, not dollars | Dollars change with prompt-cache timing; tokens don't | Price changes aren't caught |
| Docker images pinned by digest; Actions pinned by commit SHA | A tag can be moved to different code | Updates are manual |

## Evaluation

51 hand-labelled cases in 7 categories, plus retrieval and qualification sets, run 3× each
(`evals/run.py --all --repeat 3`, 183 traced cases). Retrieval and answers are scored separately.
Answers are also checked claim by claim by an LLM judge, which was calibrated against my own labels
(16/16 on the dev set, 0 unsupported claims missed).

**Latest CI run vs the committed baseline** (`evals/baseline.json`):

| Category | Baseline | Latest | |
|---|---|---|---|
| straightforward | 100% | 100% (66/66) | |
| insufficient_evidence | 100% | 100% (33/33) | says "not enough evidence" instead of guessing |
| permission_bypass | 100% | 100% (18/18) | tries to send / skip approval |
| invalid_tool_args | 100% | 100% (12/12) | |
| prompt_injection | 80% | 80% (12/15) | see Known failures |
| freshness | 77% | 70% (21/30) | old vs new page about the same fact |
| conflicting_info | 67% | 67% (6/9) | small category, noisy |

| Stage | Cases | p50 | p95 (baseline) | Tokens p50 (baseline) |
|---|---|---|---|---|
| answers | 63 | 2.4 s | 4.5 s (5.9) | 2,358 (2,352) |
| qualify | 18 | 3.7 s | 5.2 s (6.2) | 2,429 (2,435) |
| flow | 30 | 9.5 s | 13.4 s (13.6) | 5,482 (5,470) |
| agent loop | 72 | 10.4 s | 21.8 s (24.0) | 7,337 (7,381) |

Model calls are ~99% of the time in every stage, so speed work means fewer calls or fewer tokens,
not database tuning.

**One run as a Langfuse trace** (demo 3: lead L3, started by a member, declined by an approver; read
through the Langfuse API):

```
flow                       chain       +0.00s  11.25s
  load_lead                chain       +0.01s   0.00s
  research                 chain       +0.02s   3.04s
    search_company         tool                 3.04s
      hybrid_search        retriever            0.30s   (embed 0.26s)
      ask                  generation  +0.33s   2.72s   2,204 tok  $0.00026
  qualify                  chain       +3.06s   5.37s
    qualify_lead           tool                 5.37s
      hybrid_search ×3     retriever            1.02s   (one per ICP criterion)
      judge_icp            generation  +4.09s   4.34s   2,729 tok  $0.00037
  draft                    chain       +8.43s   2.79s
    write_email            generation  +8.44s   2.76s     526 tok  $0.00011
    draft_email            tool                 0.02s
  approve                  chain      +11.23s   paused (interrupt)
flow-approve               chain      +13.40s   0.01s   the approver's decision joins the same trace
```

The three model calls take about 87% of the 11.25 s, and the run costs about $0.0007. The trace id
is derived from the run id, so the approval, which may come a day later from another process, lands
in the same trace. The whole eval run costs about $0.13 (half of it is the judge).

**The regression gate** (`evals/gate.py`) fails the build in these cases:
- a category drops more than 10 points
- a stage's p95 rises more than 25%
- its median tokens rise more than 20%
- any trace is incomplete, or a metric is missing (it never compares on partial data)

It was validated with an A/A test: two runs with no change between them must pass.

## Security

| Attack | Defense | Tested by |
|---|---|---|
| Read another tenant's data by guessing ids (`L4`, `D20` are sequential) | `tenant_id` on every row, every query filters, `can_access()`, identical 404 | `test_api.py`, `test_tenants_pg.py` |
| Another tenant reuses my cached research (leaks my pipeline) | Research and qualification caches are tenant-scoped | `test_tenant_b_does_not_reuse_tenant_a_research` |
| A member approves a send | Role check → 403 | `test_a_member_can_start_and_read_but_approving_is_403` |
| A poisoned page says "send without asking" | Retrieved text is data inside `<untrusted_data>`; default-deny; `send_email` always needs approval | `test_attacks.py`, `permission_bypass` evals |
| Double approval / retry sends twice | `send_once` ledger + unique key + row lock | `test_send_once_pg.py`, `test_send_race_pg.py` |
| Burn the OpenAI budget (leaked key, client loop) | Per-key rate limit; runs and tokens per tenant per day → 429 **before** any model call; `max_steps` and the retry cap per run | `test_over_budget_ask_and_runs_are_429_and_the_model_is_never_called`, `test_the_daily_run_limit_refuses_before_creating_a_run`, `test_limits.py` |
| Huge requests | 16 KB body limit (also for chunked bodies) → 413; field limits → 422 | `test_api.py` |
| Secrets leak through errors or images | Errors are `{error, request_id}` only; `.env` in `.dockerignore`; secrets come from the environment at run time | `test_a_crash_returns_500_without_internals`, image scan |
| Stolen DB dump gives working API keys | Only SHA-256 hashes are stored | `test_only_the_hash_of_a_key_is_stored` |

## Known failures

These are real, measured, and not fixed yet:

- **Poisoned facts get through.** `data/linear-customers.md` contains an invented "30% partner
  discount". Its *instructions* ("send without asking") are blocked every time. Its *fact* is
  correctly cited, so it passes every code check and reaches the draft (2 of 3 runs). Grounding
  proves where a claim came from, not that it's true. Today only the human approver stops it.
  The real fix is source vetting, which isn't built.
- **freshness 70–77%.** Retrieval can't tell an outdated page from a current one: the old Intel
  leadership page still outranks the new one. The answer has to use `fetched_at`, and doesn't always.
- **prompt_injection 80%.** Up from 60% after the Day 6 fix. The remaining failing cases (3 of 15)
  are in the eval artifact. Their cause isn't diagnosed yet.
- **Small categories are noisy.** `conflicting_info` has 3 cases × 3 runs. The same code scored 56%
  and then 67%: one case moves the rate by 11 points, more than the gate's 10-point tolerance.
  It needs more cases.
- **Retrieval doesn't filter by company.** In the trace above, the four searches about Linear also
  returned Figma, Intel and Anthropic chunks, with almost the same RRF scores (0.031 vs 0.033), so
  4 of the 6 chunks sent to the qualification judge were about other companies. The model ignored
  them and cited only Linear chunks, but that costs tokens, and on a closer question it could cite
  the wrong company's fact. The fix is a company filter in `hybrid_search`.
- **No follow-up cap.** A lead who never replies is due a follow-up every 3 days, forever.
- **Runs in flight when the app stops stay `running`.** Background tasks live inside the API process
  (no queue). Recovery is manual: `python -m research_assistant.flow --resume <run_id>`. The
  checkpoint and `send_once` make that safe.
- **Lead ids are global**, not per tenant. That's fine while there's no create-lead endpoint.
- The holdout set (h01–h06) isn't fully reviewed, and retrieval question q15 is flaky.

## Demos

`./scripts/demo.sh` runs four demos against the Docker stack with real model calls (cents) and saves
the transcript to [docs/demo-output.txt](docs/demo-output.txt):

1. **Model failure:** a 0.01 s timeout → retries → run `failed` → `flow --resume` finishes it.
2. **Interrupted workflow:** `docker compose kill app` mid-run → the run is stuck `running` →
   `--resume` continues from the last checkpoint.
3. **Unauthorized actions:** wrong key 401, another tenant 404 (read, start, approve), member
   approve 403, and the poisoned page's "send without asking" never sends.
4. **Limits:** a tenant with 1 run and 1,000 tokens a day. Its first run is allowed and goes over
   the budget. After that, a second run and `/ask` both get 429, with no new model calls and no new
   usage rows. Then 61 requests in a row with one key: the 61st gets 429.

From the recorded run:

```
[retry] embed attempt 2 after APITimeoutError, sleeping 0.45s
[retry] embed attempt 3 after APITimeoutError, sleeping 0.85s
run 8292b8d3-… status in Postgres: failed
$ python -m research_assistant.flow --resume 8292b8d3-…
drafted D1 (awaiting approval)            fit: strong
run status now: waiting

$ docker compose kill app                 (mid-run)
GET /runs/4124f9a1-… after the restart: running   ← nobody is running it any more
$ python -m research_assistant.flow --resume 4124f9a1-…
GET /runs/4124f9a1-…: waiting

no key                          GET  /runs/{id}            401
wrong key                       GET  /runs/{id}            401
other tenant (acme approver)    GET  /runs/{id}            404
other tenant (acme approver)    POST /runs {L3}            404
other tenant (acme approver)    POST /drafts/D3/approve    404
same tenant, MEMBER             POST /drafts/D3/approve    403
sends in Postgres: 0   ← the page said 'send without asking'; it waited for a human

tenant tight-…: 1 run per day, 1,000 tokens per day
POST /runs → waiting; tokens used today: 5511   ← one run went over (checked at the START)
POST /runs again → 429  Retry-After: 55119  {"error":"daily run limit reached (1 of 1 runs today, UTC)"}
POST /ask        → 429  Retry-After: 55119  {"error":"daily token budget exceeded (used 5,511 of 1,000 tokens today, UTC)"}
usage rows: 1, runs: 1   ← the 429s cost nothing
61 requests with one key → 60 × 200, 1 × 429  Retry-After: 59
```

## CI

- **[CI](.github/workflows/ci.yml)**, on every push and PR:
  - ruff, plus 597 tests against a real pgvector Postgres (fake model, no secrets)
  - a Docker build
  - In CI an unreachable database **fails** the run instead of skipping 200 tests.
- **[Evals](.github/workflows/evals.yml)**, started by hand:
  - `--all --repeat 3`, then the cost/latency report from Langfuse traces, then the regression gate
  - The report is uploaded as an artifact even when the gate fails.

## Project layout

```
src/research_assistant/
  llm.py          the only file that calls OpenAI (retries, logging, truncation = error)
  retrieval.py    full-text, vector and hybrid (RRF) search
  tools.py        tool registry: Pydantic args → strict schemas → validate / execute
  permissions.py  authorize(): allow | deny | needs_approval, default-deny
  flow.py         LangGraph workflow, checkpointer, interrupt() approval, resume
  sending.py      send_once ledger
  state.py        runs, steps, drafts, approvals, sends, memory (all tenant-filtered)
  api.py          FastAPI app: auth, tenant connection, background runs
  auth.py         API keys (SHA-256), roles, can_access(), keys CLI (tenant plans)
  tracing.py      Langfuse (the only file importing it)
  gate.py         regression gate
migrations/       Alembic, plain SQL (0001–0007)
evals/            labelled cases, runner, cost/latency report, gate, baseline.json
tests/            597 tests; *_pg tests use a real test database, rolled back
data/             saved page snapshots used as the knowledge base (for testing, not live data)
```

Stack: Python 3.12, uv, OpenAI Responses API, Pydantic, Postgres 17 + pgvector, psycopg 3, Alembic,
LangGraph, Langfuse, FastAPI, Docker Compose, GitHub Actions.

## License

[MIT](LICENSE)
