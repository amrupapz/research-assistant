import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from research_assistant.config import Settings
from research_assistant.llm import (
    IncompleteResponseError,
    extract_function_calls,
    function_call_output,
    respond,
)
from research_assistant.prompts import AGENT_PROMPT, AGENT_PROMPT_VERSION

GET_LEAD = {
    "type": "function",
    "name": "get_lead",
    "description": "Look up a sales lead by id.",
    "parameters": {
        "type": "object",
        "properties": {"lead_id": {"type": "string"}},
        "required": ["lead_id"],
        "additionalProperties": False,
    },
    "strict": True,
}

CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_abc",
    "name": "get_lead",
    "arguments": '{"lead_id": "L1"}',
    "status": "completed",
}

PROPOSES_CALL = {
    "id": "resp_1",
    "status": "completed",
    "output": [{"type": "reasoning", "id": "rs_1", "summary": []}, CALL],
    "usage": {"input_tokens": 50, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 20},
}

FINAL = {
    "id": "resp_2",
    "status": "completed",
    "output": [
        {"type": "reasoning", "id": "rs_2", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "Dana Kim at Figma."}]},
    ],
    "usage": {"input_tokens": 80, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 10},
}


def _settings(runs_dir: Path) -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        model="test-model",
        max_output_tokens=123,
        runs_dir=runs_dir,
        base_delay_s=0.001,
        max_delay_s=0.002,
    )


def _client(*bodies: dict) -> MagicMock:
    """Fake client whose responses.create returns these bodies, one per call."""
    client = MagicMock()
    responses = []
    for body in bodies:
        r = MagicMock()
        r.model_dump.return_value = body
        r._request_id = "req_test"
        responses.append(r)
    client.responses.create.side_effect = responses
    return client


def _read_runs(runs_dir: Path) -> list[dict]:
    lines = [line for f in runs_dir.glob("*.jsonl") for line in f.read_text().splitlines()]
    return [json.loads(line) for line in lines]


def test_extract_function_calls_skips_reasoning_and_messages():
    body = {"output": [{"type": "reasoning"}, CALL, {"type": "message", "content": []}]}
    assert extract_function_calls(body) == [CALL]


def test_extract_function_calls_none_when_model_answers():
    assert extract_function_calls(FINAL) == []


def test_respond_sends_tools_and_static_instructions_without_temperature(tmp_path):
    client = _client(PROPOSES_CALL)
    items = [{"role": "user", "content": "Who is lead L1?"}]
    body = respond(items, [GET_LEAD], client=client, settings=_settings(tmp_path))

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["tools"] == [GET_LEAD]
    assert kwargs["instructions"] == AGENT_PROMPT
    assert kwargs["input"] == items
    assert kwargs["max_output_tokens"] == 123
    assert "temperature" not in kwargs
    assert body == PROPOSES_CALL


def test_respond_logs_proposed_tool_calls(tmp_path):
    respond([], [GET_LEAD], client=_client(PROPOSES_CALL), settings=_settings(tmp_path))
    [rec] = _read_runs(tmp_path)
    assert rec["kind"] == "agent_turn"
    assert rec["prompt_version"] == AGENT_PROMPT_VERSION
    assert rec["tools_offered"] == ["get_lead"]
    assert rec["tool_calls"] == ["get_lead"]
    assert rec["status"] == "completed"
    assert rec["request_id"] == "req_test"


def test_respond_incomplete_is_an_error(tmp_path):
    body = {**PROPOSES_CALL, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    with pytest.raises(IncompleteResponseError, match="max_output_tokens"):
        respond([], [GET_LEAD], client=_client(body), settings=_settings(tmp_path))
    [rec] = _read_runs(tmp_path)
    assert rec["status"] == "incomplete"


def test_round_trip_sends_back_output_items_and_matching_call_id(tmp_path):
    """The manual M0 loop: call 1 proposes, we execute, call 2 sees the result."""
    settings = _settings(tmp_path)
    client = _client(PROPOSES_CALL, FINAL)
    items: list[dict] = [{"role": "user", "content": "Who is lead L1?"}]

    first = respond(items, [GET_LEAD], client=client, settings=settings)
    items += first["output"]  # reasoning + function_call go back too
    [call] = extract_function_calls(first)
    items.append(function_call_output(call["call_id"], json.dumps({"name": "Dana Kim"})))
    second = respond(items, [GET_LEAD], client=client, settings=settings)

    sent = client.responses.create.call_args_list[1].kwargs["input"]
    assert {"type": "reasoning", "id": "rs_1", "summary": []} in sent
    assert sent[-1] == {"type": "function_call_output", "call_id": "call_abc", "output": '{"name": "Dana Kim"}'}
    assert extract_function_calls(second) == []
    assert [r["tool_calls"] for r in _read_runs(tmp_path)] == [["get_lead"], []]


def test_agent_prompt_rules():
    """v2: wait for dependent results, search by company name (LEARNING.md D3 M2). v3: send rules. v4: qualify."""
    assert AGENT_PROMPT_VERSION == "agent-v5"
    assert "lead_status" in AGENT_PROMPT and "list_followups" in AGENT_PROMPT
    assert "qualify_lead" in AGENT_PROMPT
    assert "wait for that result" in AGENT_PROMPT
    assert "never a lead id" in AGENT_PROMPT
    assert "Drafting is not sending" in AGENT_PROMPT
