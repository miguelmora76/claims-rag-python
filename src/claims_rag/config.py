from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Price(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class Settings(BaseSettings):
    """Env vars use the ASSISTANT_ prefix, e.g. ASSISTANT_PROVIDER=claude.

    provider: "fake" (offline, deterministic) or "claude" (needs ANTHROPIC_API_KEY).
    Prices are copied from published rates; verify before relying on the cost output.
    """

    model_config = SettingsConfigDict(env_prefix="ASSISTANT_")

    provider: str = "fake"
    model: str = "claude-opus-5-5"
    judge_model: str = "claude-haiku-4-5"
    max_tokens: int = 1024
    effort: str = "low"  # blank to omit; Haiku and older models reject the effort param
    top_k: int = 3
    min_score: float = 0.08
    pricing: dict[str, Price] = {
        "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
        "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_read=0.10, cache_write=1.25),
    }
