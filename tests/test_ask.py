from unittest.mock import MagicMock

from pydantic import SecretStr

from research_assistant.config import Settings
from research_assistant.llm import ask
from research_assistant.prompts import SYSTEM_PROMPT


def _settings() -> Settings:
    return Settings(openai_api_key=SecretStr("sk-test"), model="test-model", max_output_tokens=123)


def test_ask_returns_text_and_sends_system_prompt():
    client = MagicMock()
    client.responses.create.return_value.model_dump.return_value = {
        "status": "completed",
        "incomplete_details": None,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "Payments."}]},
        ],
    }

    answer = ask("What does Stripe sell?", client=client, settings=_settings())

    assert answer == "Payments."
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == SYSTEM_PROMPT  # static system prompt
    assert kwargs["input"] == "What does Stripe sell?"  # question in the user message
    assert kwargs["model"] == "test-model"
    assert kwargs["max_output_tokens"] == 123
