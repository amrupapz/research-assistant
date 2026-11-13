# Runbook

How to operate the Docker stack when something goes wrong: find out what happened, fix stuck runs,
rotate keys, and act on alerts. Every command runs from the repo root on the machine running
`docker compose`.

Commands are meant to be pasted as-is. Where a value is needed (a request id, a run id), you first
set it in a shell variable, then paste the commands that use it.

## Where things are

| What | Where | How to look |
|---|---|---|
| Request + model-call logs | `runs/YYYY-MM-DD.jsonl` in the app container (volume `runs`) | `docker compose exec app ls runs` |
| Runs, drafts, approvals, sends, usage | Postgres in the `db` container | `docker compose exec db psql -U postgres -d research_assistant` |
| Workflow position (where a run paused) | LangGraph checkpoint tables in Postgres | `GET /runs/{id}`, or resume it (below) |
| Traces (every step, timing, tokens, cost) | Langfuse, **JP region**: jp.cloud.langfuse.com → your project → Traces | search by trace id |
| Health | `GET /health` | `curl -s localhost:8000/health` |
| Alerts | `python -m research_assistant.alerts` | see [What each alert means](#what-each-alert-means) |

## First five minutes of any incident

```bash
curl -s localhost:8000/health
```
```bash
docker compose ps
```
```bash
docker compose exec app python -m research_assistant.alerts --minutes 15
```

- **`/health` gives 503 `database unavailable`:** Postgres is down or unreachable. Every request is
  failing cleanly with 503. Check `docker compose ps db` and `docker compose logs db --tail 50`, then
  `docker compose start db`. The app recovers by itself: each request opens a new connection.
- **The app container is missing or restarting:** run `docker compose logs app --tail 100`.
- **Otherwise,** the alert lines name the problem. Go to that row in the alerts table below.

## From a complaint to the cause

Every error a client sees is `{"error": "...", "request_id": "..."}`, and the same id is in the
`X-Request-ID` response header. Start from that id.

**1. Find the request in the logs.** Put the id from the client's error in a variable:

```bash
RID=5c38a35c49b242df9d0147f471ad1337
```
```bash
docker compose exec app grep -rh "$RID" runs/
```

- A `kind: api` line shows the route, status, latency, `key_id` and `tenant_id`.
- A `kind: api_error` line (status 500) has the full traceback. The client never saw it.
- If the line has a `trace_id`, go to step 3.

**2. For a workflow run, get its status and trace id:**

```bash
RUN=564c3af5-8306-40d1-aa65-464de1b8218a
```
```bash
docker compose exec db psql -U postgres -d research_assistant -c "SELECT run_id, tenant_id, status, task, started_at, finished_at, trace_id FROM runs WHERE run_id = '$RUN'"
```

**3. Open the trace in Langfuse.** Go to jp.cloud.langfuse.com → your project → **Traces**, and paste
the `trace_id` into the search box. The trace id is derived from the run id, so a run's start,
resume and approval all land in the same trace. In the trace:
- each node (`research`, `qualify`, `draft`, `approve`) with its duration
- every model call with tokens, cost, `retries` and `status`
- every tool call with the permission `decision`

A step marked ERROR is where it broke.

**4. Read the database record for the run.**
- **Workflow runs** (`flow …`): the business facts are in `drafts`, `approvals` and `sends`:

```bash
docker compose exec db psql -U postgres -d research_assistant -c "SELECT d.draft_id, d.created_at, a.decision, s.status AS send, s.sent_at FROM drafts d LEFT JOIN approvals a ON a.draft_id = d.draft_id LEFT JOIN sends s ON s.draft_id = d.draft_id WHERE d.run_id = '$RUN'"
```

- **Agent-loop runs** (`agent "task"`): every model turn and tool call is a row in `run_steps`:

```bash
docker compose exec db psql -U postgres -d research_assistant -c "SELECT seq, kind, tool, decision, approved, status FROM run_steps WHERE run_id = '$RUN' ORDER BY seq"
```

## Stuck or failed runs

**What the statuses mean:**

| Status | Meaning | Action |
|---|---|---|
| `running` for more than 5 minutes | Stuck. Its process died (container restart, crash, kill). Background runs live inside the app process, and nothing picks them up again by itself. The p95 run takes about 15 s. | Resume it |
| `failed` | Either **transient** (OpenAI timeout or 5xx after retries, DB blip) or **permanent** (a bug or bad data, recorded by the `failed` node) | Try a resume: it continues a transient failure and refuses a permanent one with a message |
| `waiting` | Normal: a draft is waiting for an approver | Nothing (approve or decline via the API) |

**Find them:**

```bash
docker compose exec db psql -U postgres -d research_assistant -c "SELECT run_id, tenant_id, status, task, started_at FROM runs WHERE (status = 'running' AND started_at < now() - interval '5 minutes') OR (status = 'failed' AND started_at > now() - interval '1 day') ORDER BY started_at"
```

**Resume one.** Set both values from the query output. Runs are tenant-scoped, so the resume has to
use the run's own tenant.

```bash
RUN=564c3af5-8306-40d1-aa65-464de1b8218a
TENANT=default
```
```bash
docker compose run --rm app python -m research_assistant.flow --resume "$RUN" --tenant "$TENANT"
```

It continues from the last checkpoint and ends `waiting` (the approver sees the draft),
`done`, or `failed`. It prints:
- `already finished (failed: a permanent error; resuming won't help)`: read the trace (above), fix
  the cause, start a new run.
- `is waiting for approval`: not stuck. An approver has to decide.

**Crash during an approval.** If the app died after an approval was accepted but before the send,
the run is `running` at the approve step. Resuming pauses it at approve **again**, and the approver
approves again. It can't send twice: `send_once` keeps a ledger row per draft with a unique key.

**A send stuck in `pending`** (the process died mid-send):

```bash
docker compose exec db psql -U postgres -d research_assistant -c "SELECT draft_id, status, idempotency_key, channel FROM sends WHERE status = 'pending'"
```

Resuming the run settles it: `send_once` takes the row lock and checks whether the email went out
(for the outbox channel: does the `.eml` file exist?). It marks the row sent if it did, and sends
otherwise. Never send by hand while a row is pending: that's the double send the ledger prevents.

## Rotating keys

Rotate on a schedule, immediately when a key may have leaked, and when someone with access leaves.
For every key, the order is: **create the new one → switch to it → check it works → revoke the old one.**
The one exception is a leaked key: **revoke first**, then fix whatever breaks.

### A customer's API key

```bash
docker compose exec app keys list-keys
```
```bash
docker compose exec app keys create-key --tenant acme --role approver --label acme-rotated
```

Give the new key to the customer. Once they've switched, revoke the old one with its id from
`list-keys`:

```bash
KEYID=3
```
```bash
docker compose exec app keys revoke-key "$KEYID"
```

Revocation takes effect on the next request (401). Keys are looked up per request; nothing caches
them. To see what a leaked key did, its `key_id` is on every access-log line:

```bash
docker compose exec app grep -rh "\"key_id\": $KEYID," runs/
```

### OpenAI API key

1. Create a new key in the OpenAI dashboard (API keys).
2. Put it in `.env` as `OPENAI_API_KEY`.
3. Restart the app so it reads the new value. `migrate` makes no model calls, so only `app` needs
   recreating:
   ```bash
   docker compose up -d --force-recreate app
   ```
4. Check it with one real `/ask` (costs a fraction of a cent). A 502 or 500 there means the key is wrong.
5. Update the GitHub secret for the evals workflow. This reads the value from your settings, so it
   never appears on screen:
   ```bash
   uv run python -c "from research_assistant.config import get_settings as g; print(g().openai_api_key.get_secret_value(), end='')" | gh secret set OPENAI_API_KEY
   ```
6. Delete the old key in the OpenAI dashboard.

### Langfuse keys

1. Langfuse (JP) → Settings → API Keys → create a new pair.
2. Put both in `.env`: `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, without quotes.
3. Recreate the app and check the new keys work (`True` = accepted):
   ```bash
   docker compose up -d --force-recreate app
   ```
   ```bash
   docker compose exec app python -c "from research_assistant.tracing import client; print(client().auth_check())"
   ```
4. Update both GitHub secrets:
   ```bash
   uv run python -c "from research_assistant.config import get_settings as g; print(g().langfuse_public_key.get_secret_value(), end='')" | gh secret set LANGFUSE_PUBLIC_KEY
   ```
   ```bash
   uv run python -c "from research_assistant.config import get_settings as g; print(g().langfuse_secret_key.get_secret_value(), end='')" | gh secret set LANGFUSE_SECRET_KEY
   ```
5. Delete the old pair in Langfuse.

A wrong Langfuse key doesn't break requests: the app keeps working, and traces fail to export with
`401` in the app log. The evals workflow does go red, at `report.py`.

### Postgres password

`POSTGRES_PASSWORD` is only read when the data volume is **first created**. Changing it in `.env`
later does nothing to the database. To rotate, change it inside Postgres first. Open psql:

```bash
docker compose exec db psql -U postgres
```

At the `postgres=#` prompt, type `\password postgres` and enter the new password twice. It's typed at
a prompt, so it never ends up in your shell history. Leave psql with `\q`. Then put the same value in `.env` as `POSTGRES_PASSWORD`, and recreate the app:

```bash
docker compose up -d --force-recreate app
```

## What each alert means

`python -m research_assistant.alerts` reads the run logs for the last `--minutes` (default 60).
It prints one `ALERT` line per problem and exits 1 if there are any. Rates only alert with at least
`--min-requests` samples (default 20). Every threshold is a flag (`--help`).

| Alert | Default threshold | Usually means | Do this |
|---|---|---|---|
| `api_error_rate` | > 5% of responses are 5xx | **503:** Postgres down. **500:** a bug (see `unhandled_errors`). **502:** the model returned unusable output | Check `/health` first. If the DB is fine, look at `unhandled_errors` and the `kind: api_error` lines |
| `unhandled_errors` | any | An exception reached the last-resort handler; the client got a bare 500 | Take the `request_id` from the alert and follow [From a complaint to the cause](#from-a-complaint-to-the-cause). The traceback is in that log line. It's a bug: write a test that reproduces it |
| `failed_jobs` | any | A background workflow raised (often an OpenAI outage after retries) | Use the `run_id` from the alert to find the run, then resume it (see [Stuck or failed runs](#stuck-or-failed-runs)) |
| `model_error_rate` | > 10% of model calls not `completed` | `error`: OpenAI timeouts, 429s or 5xx after retries. `incomplete`: answers hit `max_output_tokens`. `invalid_output` / `ungrounded`: off-schema JSON or bad citations | Check status.openai.com and your OpenAI usage limits. For `incomplete`, look at reasoning-token growth in Langfuse. For `ungrounded`, check whether a prompt or data change shipped |
| `model_p95` | > 20 s for a single call | OpenAI is slow, or prompts grew (more chunks, longer context) | Compare tokens per call in Langfuse with the eval baseline. If tokens are normal, it's the provider: wait, or lower `TIMEOUT_S` so retries happen sooner |
| `api_p95 <route>` | > 10 s on that route | `/ask` is one model call, so a slow `/ask` is usually `model_p95`. Any other route is DB time (they make no model calls) | For the DB: `docker compose exec db psql -U postgres -d research_assistant -c "SELECT pid, state, now() - query_start AS running_for, left(query, 80) FROM pg_stat_activity WHERE state <> 'idle' ORDER BY 3 DESC"` |
| `rate_limited` | > 20% of responses are 429 | A client is looping (rate limit), or a tenant ran out of runs or tokens for the day | The `429` access-log lines have `key_id` and `tenant_id`. One key in a loop: contact the owner, or revoke the key if it's abuse. A tenant at its plan limit: raise it if that's intended (`UPDATE tenants SET daily_token_budget = …`) |

**Not covered by the script:** stuck runs (status `running` for minutes) live in the database, not
the logs. Check them with the query in [Stuck or failed runs](#stuck-or-failed-runs) after any app
restart.

**Running it on a schedule:** cron every 5 minutes over the last 15. It exits 1 on alerts, and with
`SLACK_WEBHOOK_URL` set in `.env`, the ALERT lines are also posted to Slack. Add this line with
`crontab -e`, replacing the path with where the repo lives:

```
*/5 * * * * cd /path/to/research-assistant && docker compose exec -T app python -m research_assistant.alerts --minutes 15
```
