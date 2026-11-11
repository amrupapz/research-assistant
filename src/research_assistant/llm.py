"""The ONLY module that talks to OpenAI."""

import random
import re
import sys
import time
from collections.abc import Callable
from datetime import date

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import ValidationError

from research_assistant.config import Settings, get_settings
from research_assistant.log import log_run
from research_assistant.prompts import (
    AGENT_PROMPT,
    AGENT_PROMPT_VERSION,
    EMAIL_PROMPT,
    EMAIL_PROMPT_VERSION,
    PROMPT_VERSION,
    QUALIFY_PROMPT,
    QUALIFY_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_email_message,
    build_qualify_message,
    build_user_message,
)
from research_assistant.schemas import Chunk, EmailDraft, IcpJudgment, ResearchAnswer

# The exact strict schema `responses.parse(text_format=ResearchAnswer)` used to send, built by the
# SDK's own helper. We call `responses.create` instead because `.parse` validates the JSON inside
# the SDK call -- a truncated answer then raises ValidationError before our status check runs,
# and gets logged as invalid_output with no tokens or request_id (found by D3 M5 evals).
# Private SDK path: if an upgrade moves it, the import fails loudly at startup.
RESEARCH_ANSWER_FORMAT = type_to_text_format_param(ResearchAnswer)
ICP_JUDGMENT_FORMAT = type_to_text_format_param(IcpJudgment)
EMAIL_DRAFT_FORMAT = type_to_text_format_param(EmailDraft)


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
    chunks: list[Chunk],
    client: OpenAI | None = None,
    settings: Settings | None = None,
    temperature: float | None = None,
    today: date | None = None,
) -> ResearchAnswer:
    """Answer from the given chunks only; return a validated, grounded ResearchAnswer or raise.

    Always logs one run record, including which chunks were retrieved and which were cited.
    """
    settings = settings or get_settings()
    client = client or make_client(settings)
    extra = {} if temperature is None else {"temperature": temperature}  # only send if set

    record: dict = {
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "retries": 0,
        "retrieved_ids": [c.chunk_id for c in chunks],
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
            lambda: client.responses.create(  # NOT .parse: it parses JSON before anyone checks status
                model=settings.model,
                text={"format": RESEARCH_ANSWER_FORMAT},  # API forces JSON matching this schema
                instructions=SYSTEM_PROMPT,  # static -> cacheable
                input=build_user_message(question, chunks, today or date.today()),  # per-request data
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
        record["cited_ids"] = [c.chunk_id for c in result.citations]
        check_grounding(result, chunks)  # raises UngroundedAnswerError
        return result
    except KeyboardInterrupt:  # Ctrl+C is NOT an Exception: log it as interrupted, then stop
        record["status"] = "interrupted"
        raise
    except Exception as e:
        if isinstance(e, ValidationError):
            record["status"] = "invalid_output"  # API said completed, but JSON is off-schema
        if isinstance(e, UngroundedAnswerError):
            record["status"] = "ungrounded"  # valid JSON, but citations don't check out
        record.setdefault("status", "error")  # keep "incomplete" if the API told us that
        record.setdefault("request_id", getattr(e, "request_id", None))  # API errors carry it
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def judge_icp(
    company: str,
    criteria: list[tuple[str, str]],
    chunks: list[Chunk],
    client: OpenAI | None = None,
    settings: Settings | None = None,
    today: date | None = None,
) -> IcpJudgment:
    """Judge each (id, question) criterion yes/no/unknown from the chunks only. Always logs one record.

    Returns the model's raw judgment. It is NOT trusted: qualify.reconcile checks its ids and
    citations, and qualify.compute_fit (code) decides the fit.
    """
    settings = settings or get_settings()
    client = client or make_client(settings)
    record: dict = {
        "kind": "qualify_judgment",
        "model": settings.model,
        "prompt_version": QUALIFY_PROMPT_VERSION,
        "company": company,
        "retries": 0,
        "retrieved_ids": [c.chunk_id for c in chunks],
    }
    start = time.perf_counter()

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        record["retries"] += 1
        print(f"[retry] qualify attempt {attempt} after {type(exc).__name__}, sleeping {delay:.2f}s", file=sys.stderr)

    try:
        response = with_retries(
            lambda: client.responses.create(
                model=settings.model,
                text={"format": ICP_JUDGMENT_FORMAT},
                instructions=QUALIFY_PROMPT,  # static -> cacheable
                input=build_qualify_message(company, criteria, chunks, today or date.today()),
                max_output_tokens=settings.max_output_tokens,
            ),
            max_attempts=settings.max_attempts,
            base_delay=settings.base_delay_s,
            max_delay=settings.max_delay_s,
            on_retry=on_retry,
        )
        body = response.model_dump()
        record["request_id"] = response._request_id
        record["response_id"] = body.get("id")
        record["status"] = body.get("status")
        record.update(_usage(body))
        judgment = IcpJudgment.model_validate_json(extract_output_text(body))  # raises on incomplete / off-schema
        record["results"] = [{"id": r.id, "met": r.met, "chunk_ids": r.chunk_ids} for r in judgment.results]
        return judgment
    except KeyboardInterrupt:  # Ctrl+C is NOT an Exception: log it as interrupted, then stop
        record["status"] = "interrupted"
        raise
    except Exception as e:
        if isinstance(e, ValidationError):
            record["status"] = "invalid_output"
        record.setdefault("status", "error")
        record.setdefault("request_id", getattr(e, "request_id", None))
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def write_email(
    lead: dict,
    sent: list[dict],
    findings: list[dict],
    client: OpenAI | None = None,
    settings: Settings | None = None,
    today: date | None = None,
) -> EmailDraft:
    """Write an intro or follow-up from research only (D5 workflow draft node). Always logs one record.

    The result is NOT trusted: the draft node still sends it through draft_email's validation.
    """
    settings = settings or get_settings()
    client = client or make_client(settings)
    record: dict = {
        "kind": "write_email",
        "model": settings.model,
        "prompt_version": EMAIL_PROMPT_VERSION,
        "lead_id": lead.get("lead_id"),
        "retries": 0,
        "n_findings": len(findings),
        "followup": bool(sent),
    }
    start = time.perf_counter()

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        record["retries"] += 1
        print(f"[retry] email attempt {attempt} after {type(exc).__name__}, sleeping {delay:.2f}s", file=sys.stderr)

    try:
        response = with_retries(
            lambda: client.responses.create(
                model=settings.model,
                text={"format": EMAIL_DRAFT_FORMAT},
                instructions=EMAIL_PROMPT,  # static -> cacheable
                input=build_email_message(lead, sent, findings, today or date.today()),
                max_output_tokens=settings.max_output_tokens,
            ),
            max_attempts=settings.max_attempts,
            base_delay=settings.base_delay_s,
            max_delay=settings.max_delay_s,
            on_retry=on_retry,
        )
        body = response.model_dump()
        record["request_id"] = response._request_id
        record["response_id"] = body.get("id")
        record["status"] = body.get("status")
        record.update(_usage(body))
        return EmailDraft.model_validate_json(extract_output_text(body))  # raises on incomplete / off-schema
    except KeyboardInterrupt:  # Ctrl+C is NOT an Exception: log it as interrupted, then stop
        record["status"] = "interrupted"
        raise
    except Exception as e:
        if isinstance(e, ValidationError):
            record["status"] = "invalid_output"
        record.setdefault("status", "error")
        record.setdefault("request_id", getattr(e, "request_id", None))
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def respond(
    input_items: list[dict],
    tools: list[dict],
    instructions: str = AGENT_PROMPT,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict:
    """One model turn with tools. Returns the raw response body; the model only PROPOSES calls.

    The caller reads extract_function_calls(body), decides whether to run each one, and sends
    back the conversation so far + body["output"] + a function_call_output per call.
    Always logs one run record, including the names of the tools the model asked for.
    """
    settings = settings or get_settings()
    client = client or make_client(settings)

    record: dict = {
        "kind": "agent_turn",
        "model": settings.model,
        "prompt_version": AGENT_PROMPT_VERSION,
        "retries": 0,
        "tools_offered": [t["name"] for t in tools],
    }
    start = time.perf_counter()

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        record["retries"] += 1
        print(f"[retry] agent attempt {attempt} after {type(exc).__name__}, sleeping {delay:.2f}s", file=sys.stderr)

    try:
        response = with_retries(
            lambda: client.responses.create(
                model=settings.model,
                instructions=instructions,  # static -> cacheable
                input=input_items,
                tools=tools,
                max_output_tokens=settings.max_output_tokens,
            ),
            max_attempts=settings.max_attempts,
            base_delay=settings.base_delay_s,
            max_delay=settings.max_delay_s,
            on_retry=on_retry,
        )
        body = response.model_dump(exclude_none=True)  # we resend these items, so drop null fields
        record["request_id"] = response._request_id
        record["response_id"] = body.get("id")
        record["status"] = body.get("status")
        record.update(_usage(body))
        record["tool_calls"] = [c["name"] for c in extract_function_calls(body)]
        if body.get("status") != "completed":
            reason = (body.get("incomplete_details") or {}).get("reason", "unknown")
            raise IncompleteResponseError(f"Response status={body.get('status')!r}, reason={reason!r}")
        return body
    except KeyboardInterrupt:  # Ctrl+C is NOT an Exception: log it as interrupted, then stop
        record["status"] = "interrupted"
        raise
    except Exception as e:
        record.setdefault("status", "error")
        record.setdefault("request_id", getattr(e, "request_id", None))
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000)
        log_run(record, settings.runs_dir)


def extract_function_calls(body: dict) -> list[dict]:
    """The function_call items the model proposed (skips reasoning and message items)."""
    return [item for item in body.get("output", []) if item.get("type") == "function_call"]


def function_call_output(call_id: str, output: str) -> dict:
    """The input item that answers one function_call. call_id must match the call's call_id."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


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
    except KeyboardInterrupt:  # Ctrl+C is NOT an Exception: log it as interrupted, then stop
        record["status"] = "interrupted"
        raise
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


class UngroundedAnswerError(Exception):
    """The answer cites chunks we didn't give it, or makes claims without any citation."""


INLINE_CITATION = re.compile(r"\[([^\[\]\s]+#\d+)\]")  # "[figma-about#000]" -> "figma-about#000"


def check_grounding(result: ResearchAnswer, chunks: list[Chunk]) -> None:
    """Raise UngroundedAnswerError unless every citation points at a chunk we actually retrieved."""
    by_id = {c.chunk_id: c for c in chunks}
    for cit in result.citations:
        chunk = by_id.get(cit.chunk_id)
        if chunk is None:
            raise UngroundedAnswerError(f"cites {cit.chunk_id!r}, which was not retrieved")
        if cit.source_url != chunk.source_url:
            raise UngroundedAnswerError(f"{cit.chunk_id}: source_url {cit.source_url!r} != {chunk.source_url!r}")
    unknown = set(INLINE_CITATION.findall(result.answer)) - by_id.keys()
    if unknown:
        raise UngroundedAnswerError(f"answer text cites unretrieved chunks: {sorted(unknown)}")
    if not result.citations and not result.needs_sources:
        raise UngroundedAnswerError("no citations, but needs_sources is false (an uncited answer must abstain)")


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
