from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISITBRIEF_", env_file=".env", extra="ignore"
    )

    cases_dir: Path = PROJECT_ROOT / "fixtures" / "cases"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_documents: int = Field(default=3, ge=1, le=3)
    max_file_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=10 * 1024 * 1024)
    max_pages: int = Field(default=20, ge=1, le=20)
    max_extracted_chars: int = Field(default=200000, ge=1, le=200000)
    max_block_chars: int = Field(default=2000, ge=100, le=10000)
    model_backend: Literal["notebook", "token_factory"] = "notebook"
    model_base_url: AnyHttpUrl = "http://127.0.0.1:8001/v1"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    model_api_key: SecretStr = SecretStr("")
    model_timeout_seconds: float = Field(default=120, gt=0, le=300)
    inference_enabled: bool = False
    model_context_tokens: int = Field(default=8192, ge=1024, le=32768)
    model_max_output_tokens: int = Field(default=2048, ge=128, le=4096)
    # Omitted on the notebook adapter, which rejects unknown body fields.
    # DeepSeek-V4-Flash accepts "none" and then returns the JSON in content.
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] | None = None

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def blank_reasoning_effort(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def inference_configuration(self) -> "Settings":
        if self.model_max_output_tokens >= self.model_context_tokens:
            raise ValueError("Output budget must be smaller than model context")
        if self.inference_enabled:
            key = self.model_api_key.get_secret_value()
            if len(key) < 32 or not key.isascii() or any(c.isspace() for c in key):
                raise ValueError("Inference requires a private ASCII key of at least 32 characters")
            url = self.model_base_url
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("Model URL must not contain credentials, query or fragment")
            # Token Factory lives at /radeon/api/v1. The notebook adapter lives at /v1.
            if not url.path.rstrip("/").endswith("/v1"):
                raise ValueError("Model URL must end in /v1")
            if url.scheme != "https" and url.host not in {"127.0.0.1", "localhost", "[::1]"}:
                raise ValueError("Plain HTTP requires a loopback endpoint or private local tunnel")
        return self