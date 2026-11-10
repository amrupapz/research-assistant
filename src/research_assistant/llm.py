"""The ONLY module that talks to OpenAI."""

import time

from openai import OpenAI

from research_assistant.config import Settings, get_settings
from research_assistant.log import log_run
from research_assistant.prompts import PROMPT_VERSION, SYSTEM_PROMPT


def make_client(settings: Settings | None = None) -> OpenAI:
    """OpenAI client with SDK retries OFF (we build our own in M3) and an explicit timeout."""
    settings = settings or get_settings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        max_retries=0,
        timeout=settings.timeout_s,
    )


def ask(question: str, client: OpenAI | None = None, settings: Settings | None = None) -> str:
    """Ask one question; return the answer text or raise. Always logs exactly one run record."""
    settings = settings or get_settings()
    client = client or make_client(settings)

    record: dict = {"model": settings.model, "prompt_version": PROMPT_VERSION, "retries": 0}
    start = time.perf_counter()
    try:
        response = client.responses.create(
            model=settings.model,
            instructions=SYSTEM_PROMPT,  # static -> cacheable
            input=question,  # per-request data goes here
            max_output_tokens=settings.max_output_tokens,
        )
        body = response.model_dump()
        record["request_id"] = response._request_id  # x-request-id header (req_...)
        record["response_id"] = body.get("id")  # resp_...
        record["status"] = body.get("status")
        record.update(_usage(body))
        return extract_output_text(body)  # raises on incomplete/empty
    except Exception as e:
        record.setdefault("status", "error")  # keep "incomplete" if the API told us that
        record.setdefault("request_id", getattr(e, "request_id", None))  # API errors carry it
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
