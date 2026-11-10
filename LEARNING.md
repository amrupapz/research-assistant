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

