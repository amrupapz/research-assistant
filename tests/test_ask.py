import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from research_assistant.config import Settings
from research_assistant.llm import IncompleteResponseError, ask
from research_assistant.prompts import PROMPT_VERSION, SYSTEM_PROMPT


def _settings(runs_dir: Path) -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        model="test-model",
        max_output_tokens=123,
        runs_dir=runs_dir,
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


COMPLETED = {
    "id": "resp_test",
    "status": "completed",
    "incomplete_details": None,
    "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "Payments."}]},
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

    assert answer == "Payments."
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == SYSTEM_PROMPT  # static system prompt
    assert kwargs["input"] == "What does Stripe sell?"  # question in the user message
    assert kwargs["model"] == "test-model"
    assert kwargs["max_output_tokens"] == 123


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
    client.responses.create.side_effect = TimeoutError("boom")

    with pytest.raises(TimeoutError):
        ask("q", client=client, settings=_settings(tmp_path))

    [run] = _read_runs(tmp_path)
    assert run["status"] == "error"
    assert run["error"] == "TimeoutError: boom"
    assert "latency_ms" in run  # timing logged even on failure
