import pytest

from research_assistant.llm import IncompleteResponseError, extract_output_text


def _message(*texts: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": t, "annotations": []} for t in texts],
    }


def _body(status: str, output: list, incomplete_details: dict | None = None) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "status": status,
        "incomplete_details": incomplete_details,
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "error": None,
    }


def test_completed_returns_text():
    body = _body("completed", [_message("Stripe sells payments APIs.")])
    assert extract_output_text(body) == "Stripe sells payments APIs."


def test_skips_reasoning_item_that_comes_first():
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
    body = _body("completed", [reasoning, _message("Answer.")])
    assert extract_output_text(body) == "Answer."


def test_joins_multiple_output_text_parts():
    body = _body("completed", [_message("Hello, ", "world.")])
    assert extract_output_text(body) == "Hello, world."


def test_incomplete_raises_with_reason():
    body = _body(
        "incomplete",
        [_message("Stripe sells")],  # partial text present — still NOT an answer
        incomplete_details={"reason": "max_output_tokens"},
    )
    with pytest.raises(IncompleteResponseError, match="max_output_tokens"):
        extract_output_text(body)


def test_completed_but_no_message_raises():
    body = _body("completed", [{"type": "reasoning", "id": "rs_1", "summary": []}])
    with pytest.raises(IncompleteResponseError):
        extract_output_text(body)
