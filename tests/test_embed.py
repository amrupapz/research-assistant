"""llm.embed with a fake OpenAI client: no real API calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import openai
import pytest
from pydantic import SecretStr

from research_assistant.config import Settings
from research_assistant.llm import EmbeddingError, embed

REQ = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")


def _settings(runs_dir: Path) -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        model="test-model",
        embedding_model="test-embed",
        embedding_dims=3,
        runs_dir=runs_dir,
        base_delay_s=0.001,
        max_delay_s=0.002,
    )


def _response(vectors: list[list[float]], order: list[int] | None = None, tokens: int = 7):
    order = order if order is not None else list(range(len(vectors)))
    resp = MagicMock()
    resp._request_id = "req_embed"
    resp.usage.prompt_tokens = tokens
    resp.data = [SimpleNamespace(index=i, embedding=vectors[i]) for i in order]
    return resp


def _log(runs_dir: Path) -> list[dict]:
    [path] = runs_dir.glob("*.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_returns_vectors_in_input_order_and_logs(tmp_path):
    client = MagicMock()
    client.embeddings.create.return_value = _response([[1, 0, 0], [0, 1, 0]], order=[1, 0])

    assert embed(["a", "b"], client=client, settings=_settings(tmp_path)) == [[1, 0, 0], [0, 1, 0]]

    client.embeddings.create.assert_called_once_with(model="test-embed", input=["a", "b"])
    [rec] = _log(tmp_path)
    assert rec["kind"] == "embedding"
    assert rec["status"] == "completed"
    assert rec["request_id"] == "req_embed"
    assert rec["n_texts"] == 2
    assert rec["input_tokens"] == 7
    assert rec["retries"] == 0
    assert "latency_ms" in rec


def test_empty_input_makes_no_call(tmp_path):
    client = MagicMock()
    assert embed([], client=client, settings=_settings(tmp_path)) == []
    client.embeddings.create.assert_not_called()


def test_wrong_dimensions_is_an_error_and_logged(tmp_path):
    client = MagicMock()
    client.embeddings.create.return_value = _response([[1, 0]])  # 2 dims, settings say 3

    with pytest.raises(EmbeddingError, match="expected 3 dims"):
        embed(["a"], client=client, settings=_settings(tmp_path))
    [rec] = _log(tmp_path)
    assert rec["status"] == "error"
    assert rec["request_id"] == "req_embed"


def test_429_is_retried_then_succeeds(tmp_path):
    rate_limited = openai.RateLimitError(
        "slow down", response=httpx2.Response(429, request=REQ), body={"code": "rate_limit_exceeded"}
    )
    client = MagicMock()
    client.embeddings.create.side_effect = [rate_limited, _response([[1, 0, 0]])]

    assert embed(["a"], client=client, settings=_settings(tmp_path)) == [[1, 0, 0]]
    assert _log(tmp_path)[0]["retries"] == 1


def test_401_is_not_retried(tmp_path):
    auth = openai.AuthenticationError("bad key", response=httpx2.Response(401, request=REQ), body=None)
    client = MagicMock()
    client.embeddings.create.side_effect = auth

    with pytest.raises(openai.AuthenticationError):
        embed(["a"], client=client, settings=_settings(tmp_path))
    assert client.embeddings.create.call_count == 1
    assert _log(tmp_path)[0]["status"] == "error"


def test_ctrl_c_is_logged_as_interrupted_and_not_retried(tmp_path):
    client = MagicMock()
    client.embeddings.create.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        embed(["a"], client=client, settings=_settings(tmp_path))
    assert client.embeddings.create.call_count == 1
    [rec] = _log(tmp_path)
    assert rec["status"] == "interrupted"
