## D1 M0 — raw HTTP call to the Responses API

**Concept:** An LLM call is just an HTTPS POST. The SDK wraps it; seeing the raw JSON shows what we have to handle ourselves.

**Gotchas seen in real output (2026-11-10):**
- `output[0]` was a `reasoning` item, not the answer. Find items with `type == "message"`, then `output_text` parts inside them.
- Truncation (`max_output_tokens: 16`) returned **HTTP 200** with `status: "incomplete"`, `incomplete_details.reason: "max_output_tokens"`, and **no message item at all**. Reasoning used all 16 tokens. A 200 status code does not mean we got an answer.
- `usage.output_tokens` (88) includes `reasoning_tokens` (42). We pay for reasoning we never see.
- Minimum `max_output_tokens` is 16 (from the 400 error message).
- 400 and 401 are our bugs → never retry. 429 `rate_limit_exceeded` → retry with backoff. 429 `insufficient_quota` → never retry.
- Log `x-request-id` from every response; it's what OpenAI support asks for.
- `x-ratelimit-remaining-*` headers show how close we are to a 429 before it happens.
- `usage.input_tokens_details.cached_tokens` = 0 here. The prompt is far below the caching minimum, so this is expected.

**My answer:** _(fill in after check questions)_

## D1 M1 — SDK call, system prompt, config, CLI

**Concept:** The SDK sends the same POST as M0 and gives back Python objects. All OpenAI calls go through `llm.ask()`. Settings come from `.env` via pydantic-settings. The system prompt is a fixed string with a version number.

**Gotchas:**
- `max_retries=0` on the client: the SDK retries 2× by default, and that would hide the retries we build in M3.
- `SecretStr` for the key: it prints as `**********`. Call `.get_secret_value()` only where the client is built.
- `ask()` takes an optional `client`/`settings`, so tests can pass a fake client instead of patching code.
- `response.model_dump()` → the same dict shape as M0, so `extract_output_text` stays the only parser.
- The prompt telling the model to say when it isn't sure works: "hires last month" got an "I can't verify" answer, not a made-up number.

**My answer:** _(fill in after check questions)_

## D1 M2 — structured logs + runs/*.jsonl

**Concept:** Every model call writes one JSON line (who, what, how long, how many tokens, whether it worked) to `runs/YYYY-MM-DD.jsonl`. Code can search and add up these lines, which answers questions about cost, speed and failures.

**Gotchas:**
- Log in `finally`, so failed calls get a line too. A failure missing from the log is the worst kind to debug.
- Re-raise after logging. Logging must not swallow the error.
- `record.setdefault("status", "error")` keeps `"incomplete"` when the API said so: "truncated" and "crashed" are different problems.
- Two IDs: `request_id` (`req_…`, from the `x-request-id` header, the one support asks for) and `response_id` (`resp_…`, the response object's ID).
- Log lines go to stderr and the answer to stdout, so `ask "q" > answer.txt` saves just the answer.
- `cached_tokens=0` even on repeated calls: OpenAI only caches prompts of about 1024+ tokens, and ours is about 130.
- Tests write logs to pytest's `tmp_path`, never to the real `runs/`.

**My answer:** _(fill in after check questions)_

## D1 — SDK in plain terms + M1/M2 check answers (given by Claude, not mine yet)

**SDK = a library that makes the raw HTTP call for you.** M0 built the URL, headers, JSON and parsing by hand with httpx. `client.responses.create(...)` sends the same POST and turns the JSON into Python objects and exceptions (e.g. 429 → `RateLimitError`).

1. **Optional `client`:** tests pass a fake client, so no real API calls, no cost and no key needed. Without it, tests would need to patch code or would hit the API.
2. **SDK `max_retries=2` + my loop:** retries multiply. 3 SDK attempts × my 3 attempts = up to 9 requests for one timeout. The SDK's retries are also invisible to my logs, so `retries` would be wrong.
3. **Logging in `finally`:** `finally` runs on success AND on exceptions. Logging after `return` would never run on failures, and failures are the calls I most need to see.
4. **Reasoning fraction:** sum(reasoning_tokens) / sum(output_tokens) over today's `runs/YYYY-MM-DD.jsonl` (count missing values as 0).

## D1 M3 — timeouts, retries, rate limits

**Concept:** Retry only transient failures (timeout, connection, 429 rate limit, 5xx). Never retry permanent ones (400/401/404, 429 insufficient_quota, truncation). Wait longer each time (exponential backoff) plus randomness (jitter), so many clients don't retry at the same moment.

**Gotchas:**
- `max_attempts` counts total tries, not retries: 3 attempts = 1 call + 2 retries.
- Don't sleep after the final failed attempt; just raise.
- Inject `sleep` so tests run instantly and can check the delays.
- `APITimeoutError` is a subclass of `APIConnectionError`.
- 429 has two meanings: `rate_limit_exceeded` (wait and retry) vs `insufficient_quota` (out of credit, stop).

**After the week:** honor the `Retry-After` header on 429/503 instead of only our own backoff.

**My answer:** _(fill in after check questions)_

**M3 check answers (given by Claude):**
1. `max_attempts=3` → up to 3 API calls: 1 original + 2 retries. My real log showed `retries: 2` after 3 timeouts.
2. Both are HTTP 429, but `rate_limit_exceeded` = "too fast, slow down" (temporary, so waiting fixes it) and `insufficient_quota` = "out of credit" (permanent until someone adds billing, so retrying just wastes time). Decide on the error `code`, not only the status number.

## D1 M4 — structured output (responses.parse + Pydantic)

**Concept:** Send a Pydantic schema as `text_format`, and the API forces the model to reply with JSON matching it. Code can then branch on fields (`needs_sources`, `confidence`) instead of searching text for words like "verify".

**Gotchas:**
- Input tokens rose from ~130 to ~350: the schema (field names and descriptions) is sent with every request. Field descriptions act as extra prompt instructions.
- Still check `status` first (`extract_output_text`). Cut-off JSON is broken JSON.
- Validate it myself with `ResearchAnswer.model_validate_json`. If it's off-schema → status `invalid_output`, and it's NOT retried.
- `Literal["low","medium","high"]` rejects anything else ("very sure").
- Changing the prompt or schema means bumping PROMPT_VERSION (v1 → v2), so log lines show which version produced each answer.
- Real results: Stripe → high / no sources; Anthropic funding → medium / needs sources; Figma headcount → low / needs sources. A salesperson only sees the answer text, but the code can now see these fields.
- zsh: don't use `=====` as an echo separator (`=word` is expanded). Use '---'.

**My answer:** _(fill in after check questions)_

## D1 M5 — variance + seeding evals

**Concept:** Same question, same settings → the model still varies. Measure it before trusting one answer. Evals = fixed questions + expected behavior, re-run on every prompt change.

**Results (gpt-6-luna, prompt v2, 5 runs each):**
- `temperature` → 400 "not supported with this model" (a reasoning model). Not retried, which is correct for a 400. So: 5 runs at the default only.
- Stripe: 5/5 high, 5/5 needs_sources=false; 5/5 different wordings (302–374 chars).
- Anthropic latest round: 5/5 medium, 5/5 needs_sources=true; all repeat the same $13B Series F (Sept 2025) fact.
- The text varies every time, but the structured fields were 100% stable. So evals should check fields, not exact text.
- Possible prompt issue: "latest round" on facts that may be >1 year old got `medium`. Arguably it should be `low`. Day 6 eval candidate.

**Seeded:** evals/questions.jsonl, 10 questions (5 stable / 5 unknowable) with `expect_needs_sources`.

**My answer:** _(fill in after check questions)_

## D1 M6 — tests with a fake client

**Concept:** A fake client lets tests trigger failures that are hard to cause on demand (429, 401, 5xx, truncation, refusal, bad JSON) instantly and for free, so later projects can't silently break them.

**Added:** 429 then success (retries=1); 401 not retried, `request_id` taken from the error's header; refusal → IncompleteResponseError; a cut-off answer is not retried. 31 tests total, ~0.3s.

**Gotchas:**
- A refusal is logged as status `completed` + `error`: the API finished, but there was no answer. Could become its own status (`refused`) later.
- A test can be wrong too (M3: 6 timeouts with 7 attempts). Check the test's setup before changing the code.

---

## Project 1 wrap-up (Day 1)

`ask "question"` → one model call that is:
- configured from `.env` (key as SecretStr, model, timeout, limits)
- logged as one JSONL line per call, including failures (request_id, tokens, latency, status, retries, confidence)
- retried only on transient errors, with exponential backoff + jitter, and never on 400/401/404/quota/truncation
- structured: `ResearchAnswer {answer, confidence, assumptions, needs_sources}`, validated by Pydantic
- tested: 31 tests with a fake client; no real API calls in tests

**Functions I wrote:** `extract_output_text` (myself); `with_retries` (Claude gave me the code; I should be able to explain each line).

**Open questions to carry forward:**
- Over-a-year-old "latest" facts rated `medium`, not `low` → Day 6 eval.
- `cached_tokens` is always 0: prompt + schema (~350 tokens) is below the ~1024-token caching minimum.
- After the week: honor `Retry-After`; a `refused` status.


---

# Day 2 — Project 2: RAG with citations

## D2 M0 — saved pages (synthetic fixtures)

**Concept:** RAG = retrieve relevant chunks from our own sources, put them in the prompt as untrusted data, answer only from them with citations. Evaluate retrieval (was the right chunk in top-k?) separately from answers (correct + cited?) so a failure points at the right fix.

**Built:** 7 pages in `data/`, each with a `url / fetched_at / title / synthetic: true` header. The numbers are made up on purpose: if an answer contains $22.5B or 2,140, the model read our source and didn't answer from memory.
- Traps: `intel-leadership-2024.md` (outdated: same URL, old fetched_at, previous CEO); `figma-company-profile.md` (conflict: third-party estimate of 1,870 vs. 2,140 on Figma's own page).
- Distractors: Anthropic page also mentions the earlier Series G; Linear FAQ also mentions the old $16 price; Vercel changelog has 6 dated launches.
- `gold_sources` added to q06–q10 in evals/questions.jsonl, used for retrieval hit rate@k in M6.

**Answers (worked through with Claude):**
- Two Intel pages: same URL, so prefer the newer `fetched_at` (and the page's "Last updated"). If only the 2024 page is retrieved, don't state it as current: say "as of May 2024 the CEO was Pat Gelsinger; this may be out of date", with low confidence and needs_sources=true.
- Wrong Linear price: look at the retrieved chunk_ids for that request. If linear-pricing#000 wasn't in top-k, retrieval failed. If it was there but the answer says $16 (the old price in the FAQ), generation failed. → The run log must record retrieved chunk_ids (M5).

**Gotchas:**
- Synthetic ≠ real. These pages test the pipeline, not facts about the companies.
- Same URL, different fetched_at → the answer has to prefer the newest, so fetched_at matters at answer time and not only for provenance.

## D2 M1 — ingest + chunking with overlap

**Concept:** Chunks are the unit we search and cite. They have to be small enough to be about one thing and big enough to carry their context. Overlap (the next chunk repeats the last N words of the previous one) keeps a fact that lands on a boundary whole in at least one chunk.

**Built:** `ingest.py`: `parse_page` (header required, else IngestError), `chunk_text` (200-word window, 40-word overlap, slices the original text so newlines/tables survive), `chunk_page`, `load_chunks`. Stable ids `<file stem>#NNN`. `Page`/`Chunk` in schemas.py. 12 new tests, 43 total.

**Inspection (200/40):** 7 pages → 13 chunks; median 198 words, min 51. Every gold fact sits whole inside one chunk.

**Answers (worked through with Claude):**
- With overlap=0, the real boundaries split facts: Figma's competitor list breaks after "Sketch, Miro," | "Framer, Penpot, Webflow", so "Who competes with Figma?" gets half a list. Linear breaks at "Guests on the Business and Enterprise" | "plans are free", so neither half answers "Are guests free on Business?".
- A search returning figma-company-profile#001 (no headcount) is a retrieval failure. The model then has nothing to cite and should abstain (needs_sources=true). File-level gold_sources would wrongly score it a hit, because it's the right file but the wrong chunk. Fix for M6: check that the gold fact (e.g. "2,140") appears in the retrieved text, or label gold at chunk level.

**Gotchas:**
- `figma-company-profile#001` is a 51-word tail that starts mid-table ("| 9% |") and never says "Figma". A chunk loses its context. Fix in M3: embed `title + text`, not text alone.
- The overlap repeats a fact in two chunks (`$18 per user` is in linear-pricing#000 and #001). Top-k can return near-duplicates. Worth watching in M4.
- Word windows ignore structure (headings, tables). Splitting on headings is a possible upgrade, but only if the evals show it's needed.

## D2 M2 — keyword search (Postgres full-text)

**Concept:** Postgres turns each chunk into a `tsvector` (stemmed words, stopwords dropped: "raises" → `rais`) and each query into a `tsquery`. `@@` finds matches, `ts_rank_cd` ranks them, and a GIN index keeps it fast. It's strong on exact tokens (names, numbers, codes), which embeddings blur. Without it we'd be stuck with `LIKE '%x%'`: no stemming, no ranking, full table scan.

**Setup:** Upgraded Postgres 16 → 17.11 (Homebrew), pgvector 0.8.7 installed and verified (`CREATE EXTENSION vector`). DBs: `research_assistant` (dev) and `research_assistant_test`. Added `psycopg[binary]` and `database_url` in config.

**Built:** `db.py` (`init_schema`: chunks table + generated `tsv` column + GIN index; `sync_chunks`: upsert + delete stale, idempotent; caller owns the transaction). `retrieval.py` (`keyword_search`, CLI prints the tsquery). 10 new tests (53 total): 2 mock tests, 8 integration tests that run in a rolled-back transaction and skip if Postgres is down.

**Results:** Exact terms work: "Lip-Bu Tan", "Series H", "Agent Runtime", "$18" (→ `'18'`), "raised" (stemmed). Full q06/q07/q08 questions → **no matches**: `websearch_to_tsquery` ANDs every word ('much' & 'latest' & 'round' …).

**Answers (worked through with Claude, verified with real OR queries):**
- OR doesn't fix "Figma headcount". The gold chunks rank top-2 only because they repeat "Figma"; 'headcount' matched nothing (the pages say "employees"/"Figmates"). Noise appears (linear-pricing#001 via "current"). Worse: "current or CEO or Intel" ranks the **outdated 2024 page #1** (1.6 vs 1.0) because it repeats the words more. Keyword rank counts words; it knows nothing about freshness. OR trades precision for recall.
- Rollback: every test starts from the same 13 chunks, so tests stay independent. Separate test DB: sync_chunks deletes anything not in data/, so one accidental commit would wipe the dev DB (and the paid-for embeddings later). The dev DB can also hold extra rows (e.g. a manual fake#000) that make results flaky.

**Gotchas:**
- AND semantics: natural-language questions miss. Fixes: OR the terms, or let vector search (M3) handle paraphrase and fuse the two (M4).
- The title is in every chunk's tsv, so "Series H" also matches anthropic-funding#001, which only says it in the title. That's good for context but adds noise in the ranking.
- "$18" → `'18'`: the `$` is dropped, so any standalone "18" matches. ("18A" stays one token, `18a`, so it doesn't.)
- Postgres tokenizes "user/month" as a `file` token (`ts_debug`), so "Business per user" → 0 hits on the Linear page. Tokenization can silently hide the gold chunk.
- Even "Business user/month" ranks linear-pricing#001 (FAQ with the old $16) above #000 (price table). Being in the top-k doesn't mean ranked first.
- psql: a `\` meta-command takes the rest of the line as arguments. Paste it alone, never together with SQL.
- The old Postgres 16 data dir is still at /opt/homebrew/var/postgresql@16 (service stopped, unlinked). Delete later if unused.

## D2 M3 — embeddings + pgvector (vector search)

**Concept:** An embedding model maps text to a vector, and texts with similar meaning point in similar directions. pgvector stores the vectors in Postgres; `<=>` is cosine distance, so `ORDER BY embedding <=> query` returns the closest chunks. It catches paraphrases that keyword search misses ("headcount" ≈ "employees"). Its weakness is exact tokens, and it has no notion of freshness either.

**Built:** `llm.embed(texts)`: one batch, same retry rules, run-log line `kind=embedding` (request_id, n_texts, input_tokens, latency, retries), checks count + dims. `db.py`: `embedding vector(1536)` column; the upsert NULLs the embedding when title/text changes; `embed_missing` embeds only NULL rows (2nd run: embedded 0); we embed `title + "\n\n" + text`. `retrieval.vector_search` + CLI `--mode vector`. 9 new tests (62 total); the integration tests use hand-made 3-d vectors, no API calls.

**Results (text-embedding-3-small, 13 chunks = 3,162 tokens):**
| q | vector top-1 | gold in top-3? | keyword (full question) |
|---|---|---|---|
| q06 Anthropic raise | anthropic-funding#000 (.78) | ✅ | 0 hits |
| q07 Figma headcount | figma-company-profile#000 (.73), figma-about#000 #2 | ✅ both | 0 hits |
| q08 Intel CEO | **intel-leadership-2024#000** (.6653), current page #2 (.6636) | ✅ but outdated ranks #1 | 0 hits |
| q09 Linear Business | linear-pricing#000 (.62) | ✅ | 0 hits |
| q10 Vercel launch | vercel-changelog#000 (.52) | ✅ | 0 hits |

"Business per user" → Linear #001, #000 (keyword: 0 hits). "18A" → vector returns junk (Figma, .15); keyword finds all three Intel chunks.

**My answer:** _(fill in after check questions)_

**Gotchas:**
- Vector search always returns k results, even for "best pizza in Naples" (top .09). No "no match" → we need a score floor or abstain logic in M5.
- Scores are only relative: q08's outdated vs. current page differ by 0.0017. Embeddings know the topic, not which fact is current. Freshness has to come from `fetched_at` (M5).
- `figma-company-profile#001` (the short, generic table tail) shows up #3 for almost every query: a "hub" chunk that is vaguely close to everything.
- Exact tokens ("18A") → keyword wins; paraphrases → vector wins. That's the case for hybrid search (M4).
- No HNSW index yet: with 13 rows a full scan is fine. Add `USING hnsw (embedding vector_cosine_ops)` when there are thousands of rows.

## D2 M4 — hybrid search (Reciprocal Rank Fusion)

**Concept:** Keyword and vector scores are on unrelated scales (≈0.1–1.6 vs ≈0.1–0.8), so adding them lets one method silently dominate. RRF uses only rank positions: each list gives a chunk 1/(k + rank) (k=60, rank from 1), summed over the lists it appears in. Chunks both methods agree on rise to the top, and chunks only one method finds still get in.

**Built:** keyword `match="any"` (ANDs → ORs, via `CAST(... AS tsquery)`; a `::` cast isn't allowed in FROM). `hybrid_search` = top 20 any-word keyword + top 20 vector → `rrf_merge` → top k. CLI `--mode keyword|any|vector|hybrid`. 12 new tests (74 total). `rrf_merge` was written by Claude at my request. The line-by-line walkthrough is in the chat; I should be able to explain each line.

**Results (hybrid, top-1):**
| query | keyword | vector | hybrid |
|---|---|---|---|
| "18A" | intel ✅ | junk (Figma) ❌ | intel-leadership#001 ✅ |
| "Business per user" | 0 hits ❌ | linear #001 | linear #000 ✅ |
| q06 Anthropic | 0 hits | ✅ | ✅ anthropic-funding#000 |
| q07 Figma | 0 hits | profile#000 | figma-about#000 (2,140) tied with profile#000 ✅ |
| q08 Intel CEO | 0 hits | 2024 page | **2024 page still #1** ❌ |
| q09 Linear | 0 hits | ✅ | ✅ linear-pricing#000 |
| q10 Vercel | 0 hits | ✅ | ✅ vercel-changelog#000 |

**My answer:** _(fill in after check questions)_

**Gotchas:**
- RRF scores are squashed into 0.031–0.033 (the max with 2 lists is 2/61 = 0.0328). They say nothing about absolute relevance, so they can't be the M5 abstain threshold. Use the vector similarity or the LLM's judgment instead.
- Exact ties are common (figma-about#000 = figma-company-profile#000 = 1/61 + 1/62). The chunk_id tiebreak decides, which is arbitrary but deterministic.
- The outdated Intel page still wins, because both methods like it. Retrieval can't know freshness; M5 must use `fetched_at`.
- Filler at #3 (anthropic-funding#001 for "Business per user") comes from OR-mode noise plus vector always returning k.
