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

## D2 M5 — grounded answers with citations (strict)

**Concept:** The model answers only from retrieved chunks, which go in the user message inside `<untrusted_data>` (data, not instructions; this also keeps the system prompt static). Every claim cites `[chunk_id]`, and **code** checks the citations: cited ids must have been retrieved, source_url must match, and an uncited answer must abstain. Without this the model silently mixes in memory facts, invents plausible citations, and can't be audited.

**Built:** `Citation{chunk_id, source_url}` + `ResearchAnswer.citations`. Prompt **v3** (strict grounding, prefer the newest fetched_at and first-party pages, surface conflicts, flag >12-month-old evidence, abstain wording). `build_user_message(question, chunks, today)`: chunk id/url/fetched_at/title as attributes, text HTML-escaped so a page can't close `</untrusted_data>`. `check_grounding` → `UngroundedAnswerError`, logged `status="ungrounded"`, not retried. The run log gains `retrieved_ids` + `cited_ids`. The `ask` CLI = hybrid k=5 → grounded answer → sources with fetched dates. Decision: **strict**, so q01–q05 abstain (no pages for them). 12 new tests (86 total).

**Results (gpt-6-luna, v3, ~2.1k input tokens, 2–7 s each):**
| q | answer | cited | conf / needs_sources |
|---|---|---|---|
| q06 | $22.5B Series H, Oct 14 2026 ✅ (not the older Series G) | anthropic-funding#000 | high / no |
| q07 | 2,140 (Figma) vs 1,870 (third-party), trusts Figma, says they conflict ✅ | both | low / yes |
| q08 | Lip-Bu Tan ✅, although retrieval ranked the 2024 page #1 | intel-leadership#000 | high / no |
| q09 | $18 annual / $22 monthly ✅ (not the old $16) | linear-pricing#000 | high / no |
| q10 | Agent Runtime, Nov 4 2026 ✅ | vercel-changelog#000 | high / no |
| q01 Stripe | abstains ✅ (strict) | — | low / yes |
| pizza | abstains ✅ even though 5 chunks were retrieved | — | low / yes |

**My answer:** _(fill in after check questions)_

**Gotchas:**
- Freshness was solved at the **generation** step: retrieval put the 2024 Intel page #1, and the model used fetched_at to pick the 2026 page. The log proves which step did what (`retrieved_ids` vs `cited_ids`).
- q08 didn't mention that the 2024 page names a different CEO. Is a superseded snapshot a "conflict"? Debatable. Day 6 eval candidate.
- The abstain decision comes from the model, not a score threshold (RRF scores are useless for that; see M4). That's cheap here, but every off-topic question still costs a full LLM call.
- Caching (corrected): an exact repeat of the Intel question got `cached=2159`. The cache matches the prefix of the whole input (system + user message), not just the system prompt. Different questions share only system prompt + "Today/Question", which is <1024 tokens → 0. Putting the chunk block before the question would let follow-ups on the same chunks hit the cache.
- Paraphrase flips behavior: "How many people work at Figma?" retrieved the conflicting profile chunk (#3) but cited only figma-about, high / needs_sources=false. q07's wording reported the conflict. This is a generation failure that check_grounding can't catch (the citations are valid). → M6 needs paraphrase variants + a conflict check.
- `evals/variance.py` (Day 1) now abstains: `ask` requires chunks.

## D2 M6 — evals (retrieval and answers scored separately)

**Concept:** A fixed eval set with known answers, scored automatically, so a prompt or retrieval change is *measured* rather than eyeballed. Retrieval ("was the evidence in the top k?") is nearly free and deterministic. Answers are graded on structured fields + required facts, not exact wording (Day 1: text varies, fields don't).

**Built:** `evals/questions.jsonl`: 16 cases (q01–q05 abstain under strict mode, q06–q10 + paraphrases q06b–q10b, q11 off-topic), each with `expect` (answer | conflict | abstain), `gold_sources`, `gold_facts`. `evaluate.py` (pure): `retrieval_scores` (file_recall + **fact_recall**, which fixes the M1 "right file, wrong chunk" gap) and `grade_answer` (named checks). `evals/run.py --stage retrieval|answers [--only ids] [--repeat N]` → table + `runs/evals.jsonl`. 12 tests (98 total), including a test that every gold fact really is in its gold file (otherwise the eval is broken, not the system). Note: `expect_needs_sources` is a Day 1 field, now stale; it's ignored by the grader.

**Retrieval (10 cases with gold):**
| mode | @1 | @3 | @5 |
|---|---|---|---|
| keyword (AND) | 0% | 0% | 0% |
| any (OR) | 50% | 100% | 100% |
| vector | 50% | 100% | 100% |
| hybrid | **60%** | 100% | 100% |
- @1 can never pass q07 (it needs 2 files) or q08 (the 2024 page ranks #1). q10b vector@1 = right file, wrong chunk (f1.0/x0.0), so fact_recall earned its keep. k=5 is enough, and retrieval is not the bottleneck.

**Answers:**
- v3, 1 run: 16/16. **Misleading.** With `--repeat 5`: q07 5/5, **q07b 1/5** (the paraphrase drops the third-party figure; 2/8 overall incl. CLI). Retrieval passed every time → a pure generation failure.
- v4 (conflict check is now an explicit step for every question, wording-independent): q07 5/5, **q07b 5/5**; full set **16/16**.
- Side effect: q08 on v4 now mentions the 2024 page and sets needs_sources=true (v3: false). The answer is more transparent, but a superseded snapshot isn't really an unresolved conflict. The grader doesn't check needs_sources for "answer" cases, so it didn't flag this.

**My answer:** _(fill in after check questions)_

**Gotchas:**
- One run of a flaky case is a coin flip. Use `--repeat` for anything that matters, and report pass *rates*.
- A paraphrase can flip behavior on identical evidence. Always include paraphrases in the eval set.
- Evals only catch what they check: q08's needs_sources change slipped through. Day 6: add a `needs_sources` expectation per case and decide on the "superseded snapshot" policy.
- Each fix should be re-checked on the whole set, not just the failing case (q08 shifted while fixing q07b).

---

## Project 2 wrap-up (Day 2)

`ask "question"` now does hybrid retrieval → grounded, cited answer:
- `data/` → chunks (200 words / 40 overlap; stable ids; no chunk without url + fetched_at)
- Postgres 17: full-text `tsvector` + GIN, and pgvector `vector(1536)` (text-embedding-3-small, embed title+text, only new/changed chunks)
- keyword (exact tokens) + vector (paraphrase) fused with RRF; top-5
- prompt v4: answer only from `<untrusted_data>`, cite `[chunk_id]`, prefer newest/first-party, surface conflicts, abstain otherwise; `check_grounding` rejects citations to unretrieved chunks
- run log: retrieved_ids vs cited_ids, so a failure can be pinned on retrieval or generation
- evals: retrieval hit@k (100% @3 for hybrid) and answer checks (16/16 on v4, flaky case measured 5×)

**Functions:** `rrf_merge` (Claude wrote it at my request; I should be able to explain each of the 10 lines).

**Open questions to carry forward (Day 6):**
- q08: is a superseded snapshot a "conflict"? Add needs_sources expectations to the eval set.
- Abstaining costs a full LLM call; a cheap retrieval-side gate (vector similarity floor) could skip it.
- Cache: put the chunk block before the question so follow-ups on the same chunks reuse the prefix?
- Hub chunk (figma-company-profile#001) shows up everywhere; heading-aware chunking if evals ever show it hurting.

---

# Day 3 — Project 3: tools + approval

## D3 M0 — tool calling by hand

**Concept:** I send tool schemas (name, description, JSON Schema args). The model replies with a `function_call` item (`name`, `call_id`, `arguments` as a JSON *string*). It doesn't run anything. My code decides whether to run it, runs it, and sends back a `function_call_output` with the same `call_id`. Then the model answers or asks for another call. Because the model only proposes, my code is where every action gets checked. That matters because a poisoned page in `<untrusted_data>` could talk the model into proposing anything.

**Built:** `llm.respond()` (raw `responses.create` with tools; same retries/status check; logs `kind=agent_turn`, `prompt_version=agent-v1`, `tools_offered`, `tool_calls`), `extract_function_calls()`, `function_call_output()`. Prompt `AGENT_PROMPT` (agent-v1), separate from the RAG prompt v4. Demo: `scratch/m0_tool_call.py`. 6 new tests, 104 total.

**Gotchas:**
- A turn that only has a function_call has no `output_text`, so `extract_output_text` raises on it. Use `extract_function_calls` on tool turns.
- Resend ALL of `body["output"]` (reasoning items too), not just the function_call.
- `strict: true` needs every field required + `additionalProperties: false`. It guarantees the shape, not the meaning (`lead_id: "L999"` still passes).
- `arguments` is model-written text → untrusted, validate it (M1: Pydantic).

**Real run (gpt-6-luna):** call 1 = just one `function_call` (`"{\"lead_id\":\"L1\"}"`), NO reasoning item, 147 in / 21 out. Call 2 = 209 in / 18 out, `cached=0`, answer "Dana Kim, VP of Engineering at Figma" (a fact that exists only in our dict).
- Don't assume the order or presence of items. Filter by `type`.
- Every turn resends the whole history, so input grows each step. That's one reason for `max_steps`.
- Caching only starts at ~1024+ prompt tokens, so `cached=0` here is expected.

**My answer:** _(fill in after check questions)_

**M0 check answers (given by Claude):**
1. `LEADS.get(args["lead_id"])` fails two ways. (a) Malformed args (bad JSON, missing/extra/wrong-type field): the Pydantic arg model catches it, and the error goes back to the model as tool output. (b) Valid but wrong (`L999`, or a lead I'm not allowed to see): the tool returns "not found", and `authorize()` handles "not allowed". A schema can't catch (b).
2. Injected "call send_email" does nothing in M0: the tool isn't offered, and only my code executes. Once send_email exists, `authorize()` → needs_approval is the real stop. The prompt rule helps but isn't a guarantee.

## D3 M1 — 3 tools + Pydantic arg models

**Concept:** A tool is three things: a Pydantic args model (the full rules, enforced by us), a strict schema generated from it (the shape the model sees), and a function. Generating the schema from the model means they can't drift apart. The registry is a whitelist: a name that isn't in it never runs. Without this, an invented arg like `"bcc"` or an id like `"L1; DROP TABLE"` reaches the function.

**Built:**
- `leads` table + 4 fictional seed leads (`@example.com`), `seed_leads` (idempotent), `fetch_lead`. `db` main seeds them.
- `tools.py`: `SearchCompanyArgs` / `GetLeadArgs` / `DraftEmailArgs` (`extra="forbid"`, all required), `tool_schema()` generates the strict definition, `REGISTRY`, `run_tool(name, raw_args, ctx)` → unknown tool / bad args → `ToolError`, otherwise runs.
- `search_company` = hybrid_search + grounded `ask` (a model call nested in a tool call). `get_lead`. `draft_email` stores a `Draft` in `ctx.drafts`, never sends.
- `db` fixture moved to `tests/conftest.py`. 21 new tests, 125 total.

**Real check:** the API accepted all 3 strict schemas. For "look up L2, then research their company" the model proposed ONLY `get_lead {"lead_id":"L2"}`. It can't know the company until it sees the result → that's why M2 needs a loop.

**Gotchas:**
- Strict mode = shape. Keep length limits in Pydantic and send only `type`/`description`/`pattern` (safe keywords), so a schema keyword the API rejects can't cause a 400.
- `draft_email` takes the `to` address from OUR leads table, never from the model. The model can only pick a lead_id, so an injected page can't redirect mail to an arbitrary address.
- Validation errors name the field + reason but don't echo the input: the input is model-written and could carry an injection back into the next turn.
- A failed lookup must not leave side effects (no draft is stored for L999).

**My answer:** _(fill in after check questions)_

**M1 check answers (given by Claude):**
1. Three layers, each with something only it catches:
   - Strict mode (API side) stops the model from producing a wrong type, a missing field, or an extra field in the first place. Pydantic is the backup for when strict mode is off or the call comes from somewhere else.
   - Pydantic (our side) catches a body over 2000 chars or an empty subject. We don't send maxLength to the API, so strict mode allows them.
   - The tool function catches `L999`: right shape, but no such lead.
   - Next layer (M2): `authorize()` decides whether a perfectly valid call is ALLOWED.
2. Error text goes back to the model as tool output. Echoing the value would feed injected text back into the next turn, looking like a message from our own code, and would also put it in the logs. Field + reason is enough to fix the call.

## D3 M2 — agent loop

**Concept:** An agent is a loop: call the model, run what it proposes, append the results, call again, until it answers in text or hits `max_steps`. Every proposed call goes through one gate: validate → authorize → execute. Every outcome (result, error, denial) goes back as a `function_call_output` and gets its own log line. Without the loop, multi-step tasks stall. Without max_steps, a confused model loops forever. Without authorize, any valid call runs.

**Built:**
- `agent.py`: `run_agent` (loop, max_steps=6), `handle_call` (the gate), `agent "task"` CLI.
- `tools.py`: `run_tool` split into `validate_call` + `execute`, so authorize sits between them.
- `permissions.py`: `ToolCall`, `Decision`, and the `authorize` stub (✍️ mine).
- log lines `kind=tool_call` (step, call_id, tool, args≤500 chars, decision, status, latency).
- 25 new tests (12 loop + 13 for authorize, failing until I write it).

**Design decisions:**
- Status values: `invalid` (bad args/unknown tool) → never reaches authorize. `denied` / `blocked` (needs approval; M3 adds the y/n) → the tool never runs. `error` → the tool crashed.
- Fail closed: an unexpected decision value ("yes") = deny, and if authorize itself raises, the whole run stops. Better a crash than running a call nobody approved.
- Tool crash: the model gets `search_company failed (RuntimeError)`. The full error (which might hold secrets or hostnames) stays in OUR log only.
- Every call in a turn gets an output (parallel calls), or the next request is a 400.

**My answer:** _(fill in after check questions)_

**Functions:** `authorize` (Claude wrote it at my request; I should be able to explain each line):
- `ALLOWED` / `NEEDS_APPROVAL` are frozensets of exact names. Anything not in them falls through to `return "deny"`. That's default-deny: a tool someone adds later, or a name the model invents, is blocked until someone lists it on purpose.
- `in` on a set is an exact match, so "Get_Lead", "get_lead " and "send_email\n" are all denied without any special-case code.
- The checks are in order: allow first, then needs_approval, then deny. A name can't be in two sets by accident and get the looser answer.

**First real run** ("look up L2, research their company, draft an intro"): 3 turns, 4 model calls + 1 embedding.
- Step 1 had TWO calls in parallel, `get_lead` and `search_company("What does the company associated with lead L2 do?")`. The model searched before it knew the company. The search depended on the lead but was sent in the same turn.
- Retrieval hit the Intel leadership pages (probably "lead"≈"leadership"), but the QUESTION was about "lead L2's company", which no chunk mentions → `ask`: cited [], needs_sources=true. (Claude first called this a data gap. Wrong: run 2 below got the facts from the same chunks.)
- Grounding held across the tool boundary. The agent said the sources didn't cover it and wrote a generic draft instead of inventing Intel facts.
- Fix options: agent-v2 prompt ("wait for results you depend on; pass company names to search_company") vs `parallel_tool_calls=False` (guaranteed, but slower).
- The nested `ask` was the most expensive call (2,207 input tokens). Tools that call models cost a lot.

**agent-v2** (prompt: wait for results you depend on; ask search_company by company name, never lead id). Same task, 1 run:
- 4 turns, strictly one after another: `get_lead(L2)` → `search_company("What does Intel do?")` → `draft_email` → answer. Before (v1): 3 turns, search sent alongside get_lead.
- `ask` retrieved the same Intel chunks as in run 1 → now cited intel-leadership#000/#001, high, needs_sources=false. Draft names real Intel facts.
- So the run-1 failure was 100% the query. Same evidence, different question → opposite result.
- Cost: +1 turn (sequential). Worth it when there's a dependency.
- 1 run ≠ fixed. M5: repeat this case ×5, check search_company is never in step 1 and its question names the company.

**Open (for M4/M5):** the draft body has no citations and isn't checked. "Semiconductor products for PCs" paraphrases "client" products, which is fine here. But nothing stops the agent model from adding an uncited claim between search_company and draft_email. Grounding is checked inside `ask` but NOT across the tool boundary.
- agent-v2 run 2: same 4-turn sequential order, query "What does Intel do?" again (2/2). The nested `ask` had **cached=2198 / 2201** input tokens: first cache hit. Same system prompt + same question + same retrieved chunks as the previous run, and over the ~1024-token minimum. The agent turns (<1024) stay at 0. Caching cuts cost and latency; the answer doesn't change.

**M2 check answers (given by Claude):**
1. `if decision == "allow": run / else: block` fails closed: a typo, None, "yes", or a forgotten new value means the tool does NOT run. The reverse (`if == "deny": block / else: run`) fails open: any unexpected value, even "needs_approval" if someone forgets to handle it, runs the tool. Same idea as default-deny inside authorize, one layer up. Test: `test_not_allowed_means_not_executed[yes]`. The fake authorize returns "yes" → no draft stored, status=denied.
2. The next request contains both function_calls AND one function_call_output per call_id: `{"error": "no lead with id ..."}` for get_lead, plus draft_email's result or error. The failure is data the model can react to. The API requires every function_call in `input` to have a matching output (same call_id). A missing one = an incomplete conversation → 400. That's why the loop gives EVERY call an output (invalid, denied, error included). Test: `test_parallel_calls_each_get_an_output`. agent-v2's "wait for results you depend on" should stop the dependent draft_email from being sent in the same turn at all.

## D3 M3 — approval gate (send_email)

**Concept:** Some actions can't be undone (a sent email can't be recalled). For those, the program pauses and a PERSON decides, after seeing exactly what will happen. `authorize()` → `needs_approval` triggers the pause. Only an explicit yes runs the tool; anything else is a no. Without it, one injected page or model mistake sends mail in my name, and the logs only tell me afterwards.

**Built:**
- `send_email(draft_id)` writes `outbox/<id>.eml` (stdlib `EmailMessage`, From sales-agent@example.com). `outbox/` is gitignored. Never real email.
- `handle_call`: needs_approval → `approve_fn(call, ctx)` → only a literal `True` runs it. The log record gets `approved: true/false`; a no → status `rejected` and the model is told "the user did not approve; don't retry unless they ask".
- `cli_approve`: shows To/Subject/Body from OUR draft store, asks `Approve? [y/N]`. Only y/yes counts; Enter, "sure", "yes please", Ctrl-D/EOF → no. The default `no_approver` (tests/evals/batch) always says no.
- Prompt agent-v3: "Drafting is not sending. Only call send_email when the user explicitly asked."
- 24 new tests, 176 total.

**Design decisions (each one blocks a specific attack):**
- send_email takes ONLY a draft_id → the model can't change the recipient or text between my approval and the send. What I approve is what gets sent.
- The approval screen reads the draft from our store, not from the model's description of it. The model could SAY "harmless follow-up" about a draft that says something else.
- draft_id pattern `^D[0-9]{1,6}$` → no `../../` path tricks in the outbox filename.
- Subject must be one line → no "Hi\nBcc: everyone@…" header injection.
- Each draft sends once (the .eml already exists → error).
- Default = no approver = no. A run with nobody at the keyboard can never send.

**Gotcha:** the tool_call `latency_ms` for send_email now includes my thinking time at the prompt.

**My answer:** _(fill in after check questions)_

**First real run (y):** L1/Figma, 4 turns. The approval screen showed the stored draft → `y` → `outbox/D1.eml` (quoted-printable: `=E2=80=99` = ’, `=` at line end = soft break).
- Step 1 = get_lead + search_company in parallel, and that was CORRECT: the task named Figma, so the search didn't depend on the lead. agent-v2/v3 rule followed in both directions.
- Signature was blank ("Best,"): the agent has no sender identity and didn't invent one. I approved anyway, and approvers skim. Later: sender name in config.
- Agent turn 4 had cached=1066: the agent conversation itself is now over 1024 tokens.
- **Bug found by this run (fixed):** drafts live in memory, so every run restarts at D1, but outbox/ persists → the next run's D1 would hit "already sent". `_next_draft_id` now skips ids already in outbox/. Test: `test_new_run_does_not_reuse_a_sent_draft_id`. Real fix in P4: drafts in Postgres with real ids.
- Run 2 (y): got D2 (fix works), sent. I approved "andFigma Make": the source says "and Figma Make", so the model dropped a space. Facts grounded, but the typo passed the human gate. Approval ≠ proofreading. M4/M5: code checks before the human (draft facts ⊆ search_company citations, non-empty signature).
- **Run 3 (n):** `send_email → needs_approval approved=False / rejected`. outbox still only D1, D2. The model did NOT retry send_email and ended the run. ✅ The blocked action stayed blocked against the real model.
  - Its final message said "wasn't sent because the send step requires separate approval". That's misleading: it sounds pending/procedural, but I DECLINED. The tool output said "the user did not approve". M5: the final answer must say it was declined; maybe word the error as "The user declined to send D3."
  - Same task as run 2, but this time sequential (get_lead, then search). Parallel vs sequential varies run to run; both are valid here.
  - The model dropped another space ("productteams") in the final answer. Pattern across runs for gpt-6-luna.
