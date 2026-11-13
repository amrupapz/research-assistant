#!/usr/bin/env bash
# D7 M7 demos, against the Docker stack. Real model calls: costs cents, takes ~5 minutes.
#
#   ./scripts/demo.sh
#
# Needs: `docker compose up` working and data loaded (python -m research_assistant.db in the container).
# Everything is printed AND saved to docs/demo-output.txt. API keys are created for the demo, never
# printed, and revoked at the end.
#
#   1. Model failure: tiny timeout → retries → run 'failed' → resume with the normal timeout
#   2. Interrupted workflow: kill the app mid-run → run stuck 'running' → flow --resume
#   3. Unauthorized actions: 401 / 404 cross-tenant / 403 member approve; the poisoned page's
#      "send without asking" never sends
#   4. Limits: 1 run/day + 1,000 tokens/day tenant → 2nd run and /ask are 429 (no model call);
#      61 requests with one key → the 61st is 429 (rate limit). Costs one run.
set -u
cd "$(dirname "$0")/.."
mkdir -p docs
exec > >(tee docs/demo-output.txt) 2>&1

API=http://localhost:8000
section() { printf '\n\n==================== %s ====================\n' "$1"; }
say() { printf '\n$ %s\n' "$1"; }
sql() { docker compose exec -T db psql -U postgres -d research_assistant -Atc "$1"; }
app() { docker compose run --rm --no-deps -T "$@" 2>&1 | grep -v '^ Container '; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
status_of() { curl -s -H "Authorization: Bearer $2" "$API/runs/$1" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status","?"))'; }
wait_run() {  # poll until the run is no longer 'running' (max ~2 min); prints the final status
  local s
  for _ in $(seq 1 60); do
    s=$(status_of "$1" "$2")
    [ "$s" != "running" ] && break
    sleep 2
  done
  echo "$s"
}
wait_healthy() {
  for _ in $(seq 1 30); do
    curl -sf -o /dev/null "$API/health" && return 0
    sleep 2
  done
  echo "app did not become healthy" && exit 1
}
new_key() {  # new_key TENANT ROLE LABEL → sets KEY and KEY_ID; the key itself is never printed
  local out
  out=$(docker compose exec -T app keys create-key --tenant "$1" --role "$2" --label "$3")
  KEY=$(printf '%s\n' "$out" | tail -1)
  KEY_ID=$(printf '%s\n' "$out" | head -1 | sed -E 's/^key ([0-9]+).*/\1/')
}

section "0. Setup"
say "docker compose up --build -d"
docker compose up --build -d --quiet-pull 2>&1 | tail -2
wait_healthy
echo "health: $(curl -s $API/health)"
docker compose exec -T app keys add-tenant acme >/dev/null 2>&1 || true
new_key default approver demo-approver; APPROVER=$KEY; IDS="$KEY_ID"
new_key default member demo-member; MEMBER=$KEY; IDS="$IDS $KEY_ID"
new_key acme approver demo-acme; ACME=$KEY; IDS="$IDS $KEY_ID"
echo "created 3 demo keys (ids $IDS): default approver, default member, acme approver (not printed)"


section "1. Model failure → retries → failed → resumable"
echo "A 0.01 s timeout makes every OpenAI call time out. llm.py retries (backoff + jitter), then gives"
echo "up; a timeout is TRANSIENT, so the run is marked 'failed' but keeps its checkpoint."
say "docker compose run -e TIMEOUT_S=0.01 app python -m research_assistant.flow L1"
app -e TIMEOUT_S=0.01 app python -m research_assistant.flow L1 2>&1 | grep -E "^\[run\]|Timeout|flow crashed" | tail -8
RUN1=$(sql "SELECT run_id FROM runs WHERE task = 'flow L1' ORDER BY started_at DESC LIMIT 1")
echo "run $RUN1 status in Postgres: $(sql "SELECT status FROM runs WHERE run_id = '$RUN1'")"
say "python -m research_assistant.flow --resume $RUN1   (normal timeout)"
app app python -m research_assistant.flow --resume "$RUN1" 2>&1 | grep -vE "^\s*$" | head -12
echo "run status now: $(sql "SELECT status FROM runs WHERE run_id = '$RUN1'")"
D1=$(sql "SELECT 'D' || draft_id FROM drafts WHERE run_id = '$RUN1' ORDER BY draft_id DESC LIMIT 1")
if [ -n "$D1" ]; then
  say "agent approve $D1 n   (decline: keep the demo from sending)"
  app app agent approve "$D1" n 2>&1 | head -2
fi


section "2. Interrupted workflow → stuck 'running' → --resume"
say "POST /runs {lead_id: L4}"
RUN2=$(curl -s -H "Authorization: Bearer $APPROVER" -H 'content-type: application/json' -X POST "$API/runs" -d '{"lead_id":"L4"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["run_id"])')
echo "run $RUN2 started (background task inside the app)"
sleep 4
say "docker compose kill app   (SIGKILL mid-run: a crash, no graceful shutdown)"
docker compose kill app 2>&1 | tail -1
docker compose start app 2>&1 | tail -1
wait_healthy
echo "GET /runs/$RUN2 after the restart: $(status_of "$RUN2" "$APPROVER")   ← nobody is running it any more"
if [ "$(status_of "$RUN2" "$APPROVER")" = "running" ]; then
  say "python -m research_assistant.flow --resume $RUN2"
  app app python -m research_assistant.flow --resume "$RUN2" 2>&1 | grep -vE "^\s*$" | head -12
  echo "GET /runs/$RUN2: $(status_of "$RUN2" "$APPROVER")"
else
  echo "(the run finished before the kill landed: re-run the script, or raise the sleep)"
fi
D2=$(sql "SELECT 'D' || draft_id FROM drafts WHERE run_id = '$RUN2' ORDER BY draft_id DESC LIMIT 1")
[ -n "$D2" ] && app app agent approve "$D2" n >/dev/null 2>&1


section "3. Unauthorized actions are blocked"
echo "L3 is Linear. data/linear-customers.md is a poisoned page that says:"
grep -m2 -E "permission|without asking" data/linear-customers.md | sed 's/^/    > /'
say "POST /runs {lead_id: L3} as a MEMBER (members may start runs)"
RUN3=$(curl -s -H "Authorization: Bearer $MEMBER" -H 'content-type: application/json' -X POST "$API/runs" -d '{"lead_id":"L3"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["run_id"])')
echo "run $RUN3 → $(wait_run "$RUN3" "$MEMBER")"
D3=$(curl -s -H "Authorization: Bearer $MEMBER" "$API/runs/$RUN3" | python3 -c '
import sys, json
v = json.load(sys.stdin); d = v.get("draft") or {}
print(d.get("draft_id", ""))
body = d.get("body", "")
print("draft subject:", d.get("subject"), file=sys.stderr)
print("draft mentions the fake 30% discount:", "30%" in body, file=sys.stderr)')
echo "sends in Postgres: $(sql "SELECT count(*) FROM sends")   ← the page said 'send without asking'; it waited for a human"
if [ -n "$D3" ]; then
  echo
  echo "Who may do what with $D3 / run $RUN3:"
  printf '  %-58s %s\n' "no key                          GET  /runs/{id}" "$(code "$API/runs/$RUN3")"
  printf '  %-58s %s\n' "wrong key                       GET  /runs/{id}" "$(code -H 'Authorization: Bearer ra_wrong' "$API/runs/$RUN3")"
  printf '  %-58s %s\n' "other tenant (acme approver)    GET  /runs/{id}" "$(code -H "Authorization: Bearer $ACME" "$API/runs/$RUN3")"
  printf '  %-58s %s\n' "other tenant (acme approver)    GET  /leads/L3/status" "$(code -H "Authorization: Bearer $ACME" "$API/leads/L3/status")"
  printf '  %-58s %s\n' "other tenant (acme approver)    POST /runs {L3}" "$(code -H "Authorization: Bearer $ACME" -H 'content-type: application/json' -X POST "$API/runs" -d '{"lead_id":"L3"}')"
  printf '  %-58s %s\n' "other tenant (acme approver)    POST /drafts/$D3/approve" "$(code -H "Authorization: Bearer $ACME" -H 'content-type: application/json' -X POST "$API/drafts/$D3/approve" -d '{"approved":true}')"
  printf '  %-58s %s\n' "same tenant, MEMBER             POST /drafts/$D3/approve" "$(code -H "Authorization: Bearer $MEMBER" -H 'content-type: application/json' -X POST "$API/drafts/$D3/approve" -d '{"approved":true}')"
  echo
  echo "404 body for another tenant's run (identical to a run that doesn't exist):"
  echo "  $(curl -s -H "Authorization: Bearer $ACME" "$API/runs/$RUN3")"
  echo "sends in Postgres: $(sql "SELECT count(*) FROM sends")"
  say "same tenant, APPROVER: POST /drafts/$D3/approve {approved: false}   (decline)"
  echo "  $(code -H "Authorization: Bearer $APPROVER" -H 'content-type: application/json' -X POST "$API/drafts/$D3/approve" -d '{"approved":false}')"
  echo "run $RUN3 → $(wait_run "$RUN3" "$APPROVER"); sends: $(sql "SELECT count(*) FROM sends")"
fi


section "4. Limits → 429 before any model call"
STAMP=$(date +%s)
T="tight-$STAMP"
LEAD="L${STAMP: -6}"
docker compose exec -T app keys add-tenant "$T" --max-runs 1 --daily-tokens 1000 | head -1
sql "INSERT INTO leads (lead_id, name, title, company, email, tenant_id) VALUES ('$LEAD', 'Dana Kim', 'VP Engineering', 'Figma', 'dana.kim@example.com', '$T')" >/dev/null
new_key "$T" approver demo-tight; TIGHT=$KEY; IDS="$IDS $KEY_ID"
echo "tenant $T: 1 run per day, 1,000 tokens per day; one lead ($LEAD)"
HDR=$(mktemp)
show() {  # status line, Retry-After and body of one request
  local body
  body=$(curl -s -D "$HDR" "$@")
  echo "  $(head -1 "$HDR" | tr -d '\r')   Retry-After: $(grep -i '^retry-after' "$HDR" | cut -d' ' -f2 | tr -d '\r')"
  echo "  $body"
}
say "POST /runs {lead_id: $LEAD}   (under both limits: allowed)"
RUN4=$(curl -s -H "Authorization: Bearer $TIGHT" -H 'content-type: application/json' -X POST "$API/runs" -d "{\"lead_id\":\"$LEAD\"}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["run_id"])')
echo "run $RUN4 → $(wait_run "$RUN4" "$TIGHT")"
echo "tokens used today by $T: $(sql "SELECT sum(tokens) FROM usage WHERE tenant_id = '$T'")   ← one run went over 1,000 (checked at the START of a request)"
say "POST /runs again"
show -H "Authorization: Bearer $TIGHT" -H 'content-type: application/json' -X POST "$API/runs" -d "{\"lead_id\":\"$LEAD\"}"
say "POST /ask"
show -H "Authorization: Bearer $TIGHT" -H 'content-type: application/json' -X POST "$API/ask" -d '{"question":"What does Figma do?"}'
echo "usage rows for $T: $(sql "SELECT count(*) FROM usage WHERE tenant_id = '$T'"), runs: $(sql "SELECT count(*) FROM runs WHERE tenant_id = '$T'")   ← the 429s cost nothing"
D4=$(curl -s -H "Authorization: Bearer $TIGHT" "$API/runs/$RUN4" | python3 -c 'import sys,json; print((json.load(sys.stdin).get("draft") or {}).get("draft_id",""))')
[ -n "$D4" ] && curl -s -o /dev/null -H "Authorization: Bearer $TIGHT" -H 'content-type: application/json' -X POST "$API/drafts/$D4/approve" -d '{"approved":false}'
say "61 requests in a row with one fresh key (limit: 60 per minute per key)"
new_key default member demo-rate; RATE=$KEY; IDS="$IDS $KEY_ID"
CODES=$(for _ in $(seq 1 61); do code -H "Authorization: Bearer $RATE" "$API/leads/L1/status"; echo; done | sort | uniq -c | tr -s ' ' | tr '\n' ',')
echo "  status codes: $CODES"
show -H "Authorization: Bearer $RATE" "$API/leads/L1/status"
rm -f "$HDR"


section "Cleanup"
for id in $IDS; do docker compose exec -T app keys revoke-key "$id"; done
echo "saved docs/demo-output.txt"
