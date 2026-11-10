"""The ONLY module that talks to OpenAI."""

from openai import OpenAI

from research_assistant.config import Settings, get_settings
from research_assistant.prompts import SYSTEM_PROMPT


def make_client(settings: Settings | None = None) -> OpenAI:
    """OpenAI client with SDK retries OFF (we build our own in M3) and an explicit timeout."""
    settings = settings or get_settings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        max_retries=0,
        timeout=settings.timeout_s,
    )


def ask(question: str, client: OpenAI | None = None, settings: Settings | None = None) -> str:
    """Ask one question; return the answer text or raise IncompleteResponseError."""
    settings = settings or get_settings()
    client = client or make_client(settings)
    response = client.responses.create(
        model=settings.model,
        instructions=SYSTEM_PROMPT,  # static -> cacheable
        input=question,  # per-request data goes here
        max_output_tokens=settings.max_output_tokens,
    )
    return extract_output_text(response.model_dump())


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
