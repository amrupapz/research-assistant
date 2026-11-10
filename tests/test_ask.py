import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import httpx2
import openai
import pytest
from pydantic import SecretStr, ValidationError

from research_assistant.config import Settings
from research_assistant.llm import RESEARCH_ANSWER_FORMAT, IncompleteResponseError, ask
from research_assistant.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from research_assistant.schemas import Chunk, Citation, ResearchAnswer


def _settings(runs_dir: Path) -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        model="test-model",
        max_output_tokens=123,
        runs_dir=runs_dir,
        base_delay_s=0.001,  # keep retry sleeps tiny in tests
        max_delay_s=0.002,
    )


def _client(body: dict) -> MagicMock:
    """Fake OpenAI client whose responses.create returns a response with this body."""
    client = MagicMock()
    response = client.responses.create.return_value
    response.model_dump.return_value = body
    response._request_id = "req_test"
    return client


def _read_runs(runs_dir: Path) -> list[dict]:
    lines = [line for f in runs_dir.glob("*.jsonl") for line in f.read_text().splitlines()]
    return [json.loads(line) for line in lines]


CHUNKS = [
    Chunk(
        chunk_id="stripe#000",
        source_url="https://stripe.example/about",
        fetched_at=date(2026, 11, 1),
        title="About Stripe",
        text="Stripe sells payments infrastructure.",
    )
]

ANSWER_JSON = json.dumps(
    {
        "answer": "Payments. [stripe#000]",
        "confidence": "high",
        "assumptions": [],
        "needs_sources": False,
        "citations": [{"chunk_id": "stripe#000", "source_url": "https://stripe.example/about"}],
    }
)


def _with_text(body: dict, text: str) -> dict:
    """Copy of body whose message text is replaced."""
    msg = {"type": "message", "content": [{"type": "output_text", "text": text}]}
    return {**body, "output": [{"type": "reasoning", "summary": []}, msg]}


COMPLETED = {
    "id": "resp_test",
    "status": "completed",
    "incomplete_details": None,
    "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": ANSWER_JSON}]},
    ],
    "usage": {
        "input_tokens": 31,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 88,
        "output_tokens_details": {"reasoning_tokens": 42},
    },
}


def test_ask_returns_text_and_sends_system_prompt(tmp_path):
    client = _client(COMPLETED)

    answer = ask("What does Stripe sell?", CHUNKS, client=client, settings=_settings(tmp_path))

    assert answer == ResearchAnswer(
        answer="Payments. [stripe#000]",
        confidence="high",
        assumptions=[],
        needs_sources=False,
        citations=[Citation(chunk_id="stripe#000", source_url="https://stripe.example/about")],
    )
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == SYSTEM_PROMPT  # static system prompt
    assert "Question: What does Stripe sell?" in kwargs["input"]  # question + chunks in the user message
    assert '<chunk id="stripe#000"' in kwargs["input"]
    assert kwargs["model"] == "test-model"
    assert kwargs["max_output_tokens"] == 123
    assert kwargs["text"] == {"format": RESEARCH_ANSWER_FORMAT}  # schema sent to the API
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["text"]["format"]["name"] == "ResearchAnswer"


def test_success_logs_one_record_with_required_fields(tmp_path):
    ask("q", CHUNKS, client=_client(COMPLETED), settings=_settings(tmp_path))

    [run] = _read_runs(tmp_path)
    assert run["status"] == "completed"
    assert run["request_id"] == "req_test"
    assert run["response_id"] == "resp_test"
    assert run["model"] == "test-model"
    assert run["prompt_version"] == PROMPT_VERSION
    assert run["retries"] == 0
    assert (run["input_tokens"], run["output_tokens"]) == (31, 88)
    assert (run["cached_tokens"], run["reasoning_tokens"]) == (0, 42)
    assert isinstance(run["latency_ms"], int)
    assert (run["confidence"], run["needs_sources"]) == ("high", False)
    assert "error" not in run


def test_truncated_response_is_logged_as_incomplete(tmp_path):
    body = {**COMPLETED, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}

    client = _client(body)

    with pytest.raises(IncompleteResponseError):
        ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    assert client.responses.create.call_count == 1  # same request would truncate again
    [run] = _read_runs(tmp_path)
    assert run["status"] == "incomplete"
    assert "max_output_tokens" in run["error"]


def test_api_exception_is_logged_then_reraised(tmp_path):
    client = MagicMock()
    client.responses.create.side_effect = TimeoutError("boom")

    with pytest.raises(TimeoutError):
        ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    [run] = _read_runs(tmp_path)
    assert run["status"] == "error"
    assert run["error"] == "TimeoutError: boom"
    assert "latency_ms" in run  # timing logged even on failure


def test_ask_retries_timeout_and_logs_retry_count(tmp_path):
    client = _client(COMPLETED)
    good = client.responses.create.return_value
    req = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    client.responses.create.side_effect = [openai.APITimeoutError(request=req), good]

    answer = ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    assert answer.answer == "Payments. [stripe#000]"
    [run] = _read_runs(tmp_path)
    assert run["retries"] == 1
    assert run["status"] == "completed"


def test_off_schema_json_raises_logs_invalid_output_and_is_not_retried(tmp_path):
    bad = json.dumps(
        {"answer": "x", "confidence": "very sure", "assumptions": [], "needs_sources": False}
    )
    client = _client(_with_text(COMPLETED, bad))

    with pytest.raises(ValidationError):
        ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    assert client.responses.create.call_count == 1  # retrying won't fix our schema mismatch
    [run] = _read_runs(tmp_path)
    assert run["status"] == "invalid_output"
    assert run["retries"] == 0


def test_temperature_sent_only_when_set(tmp_path):
    client = _client(COMPLETED)
    ask("q", CHUNKS, client=client, settings=_settings(tmp_path))
    assert "temperature" not in client.responses.create.call_args.kwargs

    ask("q", CHUNKS, client=client, settings=_settings(tmp_path), temperature=0.2)
    assert client.responses.create.call_args.kwargs["temperature"] == 0.2
    assert _read_runs(tmp_path)[-1]["temperature"] == 0.2


# ---------- M6: failure paths through ask() with a fake client ----------

_REQ = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def _status_error(cls, status: int, code: str, request_id: str | None = None):
    headers = {"x-request-id": request_id} if request_id else {}
    response = httpx2.Response(status, request=_REQ, headers=headers)
    return cls("boom", response=response, body={"code": code})


def test_rate_limit_is_retried_then_succeeds(tmp_path):
    client = _client(COMPLETED)
    good = client.responses.create.return_value
    client.responses.create.side_effect = [
        _status_error(openai.RateLimitError, 429, "rate_limit_exceeded"),
        good,
    ]

    answer = ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    assert answer.answer == "Payments. [stripe#000]"
    assert client.responses.create.call_count == 2
    [run] = _read_runs(tmp_path)
    assert (run["status"], run["retries"]) == ("completed", 1)


def test_bad_key_401_not_retried_and_request_id_logged(tmp_path):
    client = MagicMock()
    client.responses.create.side_effect = _status_error(
        openai.AuthenticationError, 401, "invalid_api_key", request_id="req_401"
    )

    with pytest.raises(openai.AuthenticationError):
        ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    assert client.responses.create.call_count == 1
    [run] = _read_runs(tmp_path)
    assert run["status"] == "error"
    assert run["retries"] == 0
    assert run["request_id"] == "req_401"  # failed calls still carry the support ID


def test_refusal_raises_instead_of_returning_empty_answer(tmp_path):
    refusal = {"type": "message", "content": [{"type": "refusal", "refusal": "I can't help."}]}
    client = _client({**COMPLETED, "output": [refusal]})

    with pytest.raises(IncompleteResponseError):
        ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    [run] = _read_runs(tmp_path)
    assert run["status"] == "completed"  # API finished fine...
    assert "no output_text" in run["error"]  # ...but there was no answer to use


def test_cut_off_json_is_incomplete_not_invalid_output(tmp_path):
    """D3 M5 eval bug: with responses.parse the SDK raised ValidationError on half-written JSON
    before our status check, so truncation was logged as invalid_output with no tokens/request_id."""
    body = _with_text(
        {**COMPLETED, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        '{\n  "answer": "Intel builds chips for PCs, data center, and AI. Its product priorities, not',
    )
    client = _client(body)

    with pytest.raises(IncompleteResponseError, match="max_output_tokens"):
        ask("q", CHUNKS, client=client, settings=_settings(tmp_path))

    assert not client.responses.parse.called  # the SDK never gets a chance to parse first
    [run] = _read_runs(tmp_path)
    assert run["status"] == "incomplete"
    assert run["request_id"] == "req_test"
    assert run["output_tokens"] == 88  # usage is logged, so we can SEE the budget ran out
