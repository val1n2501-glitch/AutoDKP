import pytest
from pydantic import ValidationError

from autodkp.config import Settings


def test_allowed_ids_are_parsed_from_csv() -> None:
    settings = Settings(
        _env_file=None,
        allowed_telegram_ids="101, 202",  # type: ignore[arg-type]
    )
    assert settings.allowed_telegram_ids == frozenset({101, 202})


def test_invalid_allowed_ids_fail_closed() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(_env_file=None, allowed_telegram_ids="101,not-an-id")  # type: ignore[arg-type]
