"""Settings loaded from .env (and real environment variables, which win)."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr  # SecretStr: prints as '**********', never the real key
    model: str
    timeout_s: float = 30.0
    max_output_tokens: int = 1000


@lru_cache
def get_settings() -> Settings:
    """Load settings once; tests can call get_settings.cache_clear()."""
    return Settings()
