"""The ONLY module that talks to OpenAI."""

import random
import sys
import time
from collections.abc import Callable

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import ValidationError

from research_assistant.config import Settings, get_settings
from research_assistant.log import log_run
from research_assistant.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from research_assistant.schemas import ResearchAnswer


def make_client(settings: Settings | None = None) -> OpenAI:
    """OpenAI client with SDK retries OFF (we build our own in M3) and an explicit timeout."""
    settings = settings or get_settings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        max_retries=0,
        timeout=settings.timeout_s,
    )


def is_retryable(exc: BaseException) -> bool:
    """True only for failures that might succeed if we try again later."""
    if isinstance(exc, RateLimitError):
        return exc.code != "insufficient_quota"  # out of credit: waiting won't help
    if isinstance(exc, APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500  # server's fault; 4xx is ours -> never retry
    return False  # anything else (incl. IncompleteResponseError) is not transient


def with_retries[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    on_retry: Callable[[int, BaseException, float], None] = lambda attempt, exc, delay: None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(); on a retryable error, back off and try again, up to max_attempts total.

    ✍️ YOU WRITE THIS (Project 1 core function #2).

    Contract:
      - attempt counts from 1. Call fn(); if it returns, return its value.
      - If it raises:
          * not is_retryable(exc)        -> re-raise immediately (bare `raise`)
          * attempt == max_attempts      -> re-raise (out of tries)
          * otherwise: delay = backoff with jitter (below),
                       on_retry(attempt + 1, exc, delay)   # the attempt we're ABOUT to make
                       sleep(delay), then loop.
      - Backoff with jitter for the retry after attempt n:
            cap   = min(max_delay, base_delay * 2 ** (n - 1))   # 0.5, 1, 2, 4, 8, 8...
            delay = random.uniform(cap / 2, cap)                # "equal jitter"

    Hints:
      - `for attempt in range(1, max_attempts + 1):` with try/except inside.
      - Use the `sleep` argument, never time.sleep directly — tests pass a fake.
      - ~12 lines. Run: uv run pytest tests/test_retries.py -v
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            cap = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = random.uniform(cap / 2, cap)
            on_retry(attempt + 1, exc, delay)
            sleep(delay)
    raise ValueError("max_attempts must be >= 1")



def ask(
    question: str,
    client: OpenAI | None = None,
    settings: Settings | None = None,
    temperature: float | None = None,
) -> ResearchAnswer:
    """Ask one question; return a validated ResearchAnswer or raise. Always logs one run record."""
    settings = settings or get_settings()
    client = client or make_client(settings)
    extra = {} if temperature is None else {"temperature": temperature}  # only send if set

    record: dict = {
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "retries": 0,
    }
    start = time.perf_counter()

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        record["retries"] += 1
        print(
            f"[retry] attempt {attempt} after {type(exc).__name__}, sleeping {delay:.2f}s",
            file=sys.stderr,
        )

    try:
        response = with_retries(
            lambda: client.responses.parse(
                model=settings.model,
                text_format=ResearchAnswer,  # API forces JSON matching this schema
                instructions=SYSTEM_PROMPT,  # static -> cacheable
                input=question,  # per-request data goes here
                max_output_tokens=settings.max_output_tokens,
                **extra,
            ),
            max_attempts=settings.max_attempts,
            base_delay=settings.base_delay_s,
            max_delay=settings.max_delay_s,
            on_retry=on_retry,
        )
        body = response.model_dump()
        record["request_id"] = response._request_id  # x-request-id header (req_...)
        record["response_id"] = body.get("id")  # resp_...
        record["status"] = body.get("status")
        record.update(_usage(body))
        text = extract_output_text(body)  # raises on incomplete/empty
        result = ResearchAnswer.model_validate_json(text)  # raises ValidationError if off-schema
        record["confidence"] = result.confidence
        record["needs_sources"] = result.needs_sources
        return result
    except Exception as e:
        if isinstance(e, ValidationError):
            record["status"] = "invalid_output"  # API said completed, but JSON is off-schema
        record.setdefault("status", "error")  # keep "incomplete" if the API told us that
        record.setdefault("request_id", getattr(e, "request_id", None))  # API errors carry it
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


class EmbeddingError(Exception):
    """The embeddings response doesn't match what we asked for (count or dimensions)."""


def embed(
    texts: list[str],
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed texts in one batch; return one vector per text, in input order. Always logs one run record."""
    if not texts:
        return []
    settings = settings or get_settings()
    client = client or make_client(settings)

    record: dict = {"kind": "embedding", "model": settings.embedding_model, "n_texts": len(texts), "retries": 0}
    start = time.perf_counter()

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        record["retries"] += 1
        print(f"[retry] embed attempt {attempt} after {type(exc).__name__}, sleeping {delay:.2f}s", file=sys.stderr)

    try:
        response = with_retries(
            lambda: client.embeddings.create(model=settings.embedding_model, input=texts),
            max_attempts=settings.max_attempts,
            base_delay=settings.base_delay_s,
            max_delay=settings.max_delay_s,
            on_retry=on_retry,
        )
        record["request_id"] = response._request_id
        record["input_tokens"] = response.usage.prompt_tokens
        vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        bad = {len(v) for v in vectors} - {settings.embedding_dims}
        if bad:
            raise EmbeddingError(f"expected {settings.embedding_dims} dims, got {sorted(bad)}")
        record["status"] = "completed"
        return vectors
    except Exception as e:
        record["status"] = "error"
        record.setdefault("request_id", getattr(e, "request_id", None))
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def _usage(body: dict) -> dict:
    """Pull token counts out of a Responses API body (missing -> None)."""
    usage = body.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
    }


class IncompleteResponseError(Exception):
    """The model did not produce a complete answer (truncated, failed, or empty)."""


def extract_output_text(body: dict) -> str:
    """Return the assistant's text from a raw Responses API JSON body.

    ✍️ YOU WRITE THIS (Project 1 core function #1).

    Contract:
      - If body["status"] != "completed": raise IncompleteResponseError.
        Include the reason from body["incomplete_details"] in the message if present.
      - Otherwise, walk body["output"] (a list). Only items with type == "message"
        hold the answer; skip everything else (e.g. type == "reasoning").
      - Inside each message item, item["content"] is a list; collect the "text" of
        every part with type == "output_text", joined in order with "".
      - If you found no text at all: raise IncompleteResponseError.

    Hints:
      - The answer is NOT always output[0] — a reasoning item often comes first.
      - incomplete_details may be None; use .get() and guard for that.
      - ~10–15 lines. Run: uv run pytest tests/test_llm.py -v
    """
    status = body.get("status")                                   # 1
    if status != "completed":                                     # 2
        details = body.get("incomplete_details") or {}            # 3
        reason = details.get("reason", "unknown")                 # 4
        raise IncompleteResponseError(                            # 5
            f"Response status={status!r}, reason={reason!r}"
        )

    parts: list[str] = []                                         # 6
    for item in body.get("output", []):                           # 7
        if item.get("type") != "message":                         # 8
            continue
        for part in item.get("content", []):                      # 9
            if part.get("type") == "output_text":                 # 10
                parts.append(part["text"])                        # 11

    text = "".join(parts)                                         # 12
    if not text:                                                  # 13
        raise IncompleteResponseError("Response completed but contained no output_text")
    return text                                                   # 14
