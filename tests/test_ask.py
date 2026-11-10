import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr, ValidationError

from research_assistant.config import Settings
from research_assistant.llm import IncompleteResponseError, ask
from research_assistant.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from research_assistant.schemas import ResearchAnswer


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
    response = client.responses.parse.return_value
    response.model_dump.return_value = body
    response._request_id = "req_test"
    return client


def _read_runs(runs_dir: Path) -> list[dict]:
    lines = [line for f in runs_dir.glob("*.jsonl") for line in f.read_text().splitlines()]
    return [json.loads(line) for line in lines]


ANSWER_JSON = json.dumps(
    {
        "answer": "Payments.",
        "confidence": "high",
        "assumptions": [],
        "needs_sources": False,
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

    answer = ask("What does Stripe sell?", client=client, settings=_settings(tmp_path))

    assert answer == ResearchAnswer(
        answer="Payments.", confidence="high", assumptions=[], needs_sources=False
    )
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["instructions"] == SYSTEM_PROMPT  # static system prompt
    assert kwargs["input"] == "What does Stripe sell?"  # question in the user message
    assert kwargs["model"] == "test-model"
    assert kwargs["max_output_tokens"] == 123
    assert kwargs["text_format"] is ResearchAnswer  # schema sent to the API


def test_success_logs_one_record_with_required_fields(tmp_path):
    ask("q", client=_client(COMPLETED), settings=_settings(tmp_path))

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

    with pytest.raises(IncompleteResponseError):
        ask("q", client=_client(body), settings=_settings(tmp_path))

    [run] = _read_runs(tmp_path)
    assert run["status"] == "incomplete"
    assert "max_output_tokens" in run["error"]


def test_api_exception_is_logged_then_reraised(tmp_path):
    client = MagicMock()
    client.responses.parse.side_effect = TimeoutError("boom")

    with pytest.raises(TimeoutError):
        ask("q", client=client, settings=_settings(tmp_path))

    [run] = _read_runs(tmp_path)
    assert run["status"] == "error"
    assert run["error"] == "TimeoutError: boom"
    assert "latency_ms" in run  # timing logged even on failure


def test_ask_retries_timeout_and_logs_retry_count(tmp_path):
    import httpx2
    import openai

    client = _client(COMPLETED)
    good = client.responses.parse.return_value
    req = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    client.responses.parse.side_effect = [openai.APITimeoutError(request=req), good]

    answer = ask("q", client=client, settings=_settings(tmp_path))

    assert answer.answer == "Payments."
    [run] = _read_runs(tmp_path)
    assert run["retries"] == 1
    assert run["status"] == "completed"


def test_off_schema_json_raises_logs_invalid_output_and_is_not_retried(tmp_path):
    bad = json.dumps(
        {"answer": "x", "confidence": "very sure", "assumptions": [], "needs_sources": False}
    )
    client = _client(_with_text(COMPLETED, bad))

    with pytest.raises(ValidationError):
        ask("q", client=client, settings=_settings(tmp_path))

    assert client.responses.parse.call_count == 1  # retrying won't fix our schema mismatch
    [run] = _read_runs(tmp_path)
    assert run["status"] == "invalid_output"
    assert run["retries"] == 0
