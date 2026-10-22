from pathlib import Path

import pytest

from autodkp.cli import configure_bot, configure_ocr, rotate_bot_token
from autodkp.config import Settings


def test_configure_bot_writes_private_env_without_printing_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".env"
    token = "123456789:" + "A" * 35
    monkeypatch.setattr("getpass.getpass", lambda _: token)
    monkeypatch.setattr("builtins.input", lambda _: "101")

    configure_bot(env_file)

    assert env_file.stat().st_mode & 0o777 == 0o600
    assert token not in capsys.readouterr().out
    settings = Settings(_env_file=env_file)
    assert settings.telegram_bot_token is not None
    assert settings.telegram_bot_token.get_secret_value() == token
    assert settings.allowed_telegram_ids == frozenset({101})


def test_configure_bot_refuses_to_overwrite_existing_secret(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("existing=true\n")
    with pytest.raises(RuntimeError, match="already exists"):
        configure_bot(env_file)


@pytest.mark.parametrize(
    "token",
    ["short", "1234:" + "A" * 35, "123456:contains spaces"],
)
def test_configure_bot_rejects_invalid_token_before_asking_for_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda _: token)
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("user ID prompt must not be reached")
    )
    with pytest.raises(ValueError, match="token format"):
        configure_bot(tmp_path / ".env")


def test_rotate_bot_token_preserves_id_and_private_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    old_token = "123456789:" + "A" * 35
    new_token = "987654321:" + "B" * 35
    env_file.write_text(
        f"TELEGRAM_BOT_TOKEN={old_token}\nALLOWED_TELEGRAM_IDS=101\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("getpass.getpass", lambda _: new_token)

    rotate_bot_token(env_file)

    assert env_file.stat().st_mode & 0o777 == 0o600
    settings = Settings(_env_file=env_file)
    assert settings.telegram_bot_token is not None
    assert settings.telegram_bot_token.get_secret_value() == new_token
    assert settings.allowed_telegram_ids == frozenset({101})


def test_configure_ocr_adds_key_without_exposing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=123456789:" + "A" * 35 + "\n")
    api_key = "sk-test-" + "B" * 32
    monkeypatch.setattr("getpass.getpass", lambda _: api_key)

    configure_ocr(env_file)

    assert env_file.stat().st_mode & 0o777 == 0o600
    assert api_key not in capsys.readouterr().out
    settings = Settings(_env_file=env_file)
    assert settings.ocr_provider == "openai"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == api_key
