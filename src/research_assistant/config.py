"""Settings loaded from .env (and real environment variables, which win)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr  # SecretStr: prints as '**********', never the real key
    model: str
    timeout_s: float = 30.0
    max_output_tokens: int = 2000  # includes REASONING tokens; 1000 truncated ask (D3 M5)
    runs_dir: Path = Path("runs")
    max_attempts: int = 3  # total tries, including the first
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    database_url: str = "postgresql://localhost/research_assistant"
    embedding_model: str = "text-embedding-3-small"  # EMBEDDING_MODEL in .env wins
    embedding_dims: int = 1536  # must match the model; checked on every embed call
    judge_model: str | None = None  # D6 M3 claim checker: JUDGE_MODEL in .env, stronger than MODEL

    rate_limit_per_minute: int = 60  # D7 M3: requests per API key per minute (in-memory, per process)

    # D6 M1 tracing (Langfuse Cloud). Off unless TRACING_ENABLED=true AND both keys are set.
    tracing_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"  # EU; US is https://us.cloud.langfuse.com

    # USD per 1M tokens, by model name. Langfuse can't price a model it doesn't know, so we send
    # cost ourselves. A model missing here gets tokens but no cost. Override with PRICES='{...}' (JSON).
    # Keys: input (uncached), cached_input, output (reasoning tokens are billed as output and are
    # already inside output_tokens: don't add them twice). No cached_input -> billed at input.
    # Source: third-party pricing guides, Nov 2026 (check openai.com/api/pricing). GPT-6 prompts
    # over ~272K input tokens cost more; ours are ~2-3K.
    prices: dict[str, dict[str, float]] = {
        "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
        "gpt-6.1-terra": {"input": 1.00, "output": 5.00},  # JUDGE_MODEL (M3); cached price not listed
        "text-embedding-3-small": {"input": 0.02},
    }


@lru_cache
def get_settings() -> Settings:
    """Load settings once; tests can call get_settings.cache_clear()."""
    return Settings()
