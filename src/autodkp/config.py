from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_bot_token: SecretStr | None = None
    allowed_telegram_ids: frozenset[int] = Field(default_factory=frozenset)
    template_path: Path = Path("resources/templates/dkp-agreement_new_logo.pdf")
    pdf_font_path: Path = Path(
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    )
    deal_ttl_minutes: int = Field(default=60, ge=5, le=24 * 60)
    ocr_provider: str = "disabled"
    openai_api_key: SecretStr | None = None
    openai_ocr_model: str = "gpt-5.5"
    ocr_max_file_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    ocr_max_image_pixels: int = Field(default=25_000_000, ge=1_000_000)
    log_level: str = "INFO"

    @field_validator("ocr_provider")
    @classmethod
    def validate_ocr_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"disabled", "openai"}:
            raise ValueError("OCR_PROVIDER must be disabled or openai")
        return normalized

    @field_validator("allowed_telegram_ids", mode="before")
    @classmethod
    def parse_allowed_ids(cls, value: object) -> object:
        if isinstance(value, int):
            return frozenset({value})
        if isinstance(value, str):
            if not value.strip():
                return frozenset()
            return frozenset(int(item.strip()) for item in value.split(","))
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return normalized

