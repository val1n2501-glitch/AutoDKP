from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import re
import secrets
import sys
import tempfile
from hmac import compare_digest
from pathlib import Path

from autodkp.bot.app import run_bot
from autodkp.config import Settings
from autodkp.domain.demo import make_demo_deal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autodkp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo-pdf", help="Generate a synthetic agreement")
    demo.add_argument("--output", type=Path, default=Path("output/pdf/demo-dkp.pdf"))
    subparsers.add_parser("bot", help="Run the Telegram bot using long polling")
    configure = subparsers.add_parser(
        "configure-bot", help="Securely create the local .env for Telegram"
    )
    configure.add_argument("--env-file", type=Path, default=Path(".env"))
    pair = subparsers.add_parser(
        "pair-bot",
        help="Securely configure Telegram by pairing the first user with a one-time code",
    )
    pair.add_argument("--env-file", type=Path, default=Path(".env"))
    pair.add_argument("--timeout", type=int, default=300)
    rotate = subparsers.add_parser(
        "rotate-bot-token",
        help="Securely replace only the Telegram token in an existing .env",
    )
    rotate.add_argument("--env-file", type=Path, default=Path(".env"))
    rotate.add_argument("--token-stdin", action="store_true", help=argparse.SUPPRESS)
    ocr = subparsers.add_parser(
        "configure-ocr", help="Securely add an OpenAI API key to an existing .env"
    )
    ocr.add_argument("--env-file", type=Path, default=Path(".env"))
    ocr.add_argument("--key-stdin", action="store_true", help=argparse.SUPPRESS)
    return parser


_BOT_TOKEN_PATTERN = re.compile(r"^[1-9][0-9]{4,14}:[A-Za-z0-9_-]{30,}$")


def _write_configuration(env_file: Path, token: str, user_id: int) -> Path:
    if env_file.exists():
        raise RuntimeError(f"Configuration already exists: {env_file}")
    content = (
        f"TELEGRAM_BOT_TOKEN={token}\n"
        f"ALLOWED_TELEGRAM_IDS={user_id}\n"
        "TEMPLATE_PATH=resources/templates/dkp-agreement_new_logo.pdf\n"
        "PDF_FONT_PATH=/System/Library/Fonts/Supplemental/Times New Roman.ttf\n"
        "DEAL_TTL_MINUTES=60\n"
        "LOG_LEVEL=INFO\n"
    )
    env_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{env_file.name}-", dir=env_file.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, env_file)
        return env_file
    finally:
        temporary.unlink(missing_ok=True)


def _read_token(*, from_stdin: bool = False) -> str:
    token = (
        sys.stdin.readline().strip()
        if from_stdin
        else getpass.getpass("Telegram bot token (hidden): ").strip()
    )
    if not _BOT_TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid Telegram bot token format")
    return token


def configure_bot(env_file: Path) -> Path:
    if env_file.exists():
        raise RuntimeError(f"Configuration already exists: {env_file}")
    token = _read_token()
    raw_user_id = input("Allowed numeric Telegram user ID: ").strip()
    if not raw_user_id.isascii() or not raw_user_id.isdecimal() or int(raw_user_id) <= 0:
        raise ValueError("Telegram user ID must be a positive ASCII integer")
    return _write_configuration(env_file, token, int(raw_user_id))


def rotate_bot_token(env_file: Path, *, token_from_stdin: bool = False) -> Path:
    if not env_file.is_file():
        raise RuntimeError(f"Configuration does not exist: {env_file}")
    token = _read_token(from_stdin=token_from_stdin)
    lines = env_file.read_text(encoding="utf-8").splitlines()
    token_indexes = [
        index for index, line in enumerate(lines) if line.startswith("TELEGRAM_BOT_TOKEN=")
    ]
    if len(token_indexes) != 1:
        raise RuntimeError("Configuration must contain exactly one TELEGRAM_BOT_TOKEN")
    lines[token_indexes[0]] = f"TELEGRAM_BOT_TOKEN={token}"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{env_file.name}-", dir=env_file.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, env_file)
        return env_file
    finally:
        temporary.unlink(missing_ok=True)


def configure_ocr(env_file: Path, *, key_from_stdin: bool = False) -> Path:
    if not env_file.is_file():
        raise RuntimeError(f"Configuration does not exist: {env_file}")
    api_key = (
        sys.stdin.readline().strip()
        if key_from_stdin
        else getpass.getpass("OpenAI API key (hidden): ").strip()
    )
    if (
        not api_key.startswith("sk-")
        or len(api_key) < 20
        or any(char.isspace() for char in api_key)
    ):
        raise ValueError("Invalid OpenAI API key format")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    updates = {
        "OCR_PROVIDER": "openai",
        "OPENAI_API_KEY": api_key,
        "OPENAI_OCR_MODEL": "gpt-5.5",
    }
    retained = [
        line for line in lines if line.partition("=")[0] not in updates
    ]
    retained.extend(f"{key}={value}" for key, value in updates.items())
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{env_file.name}-", dir=env_file.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(retained) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, env_file)
        return env_file
    finally:
        temporary.unlink(missing_ok=True)


async def pair_bot(env_file: Path, timeout_seconds: int) -> Path:
    from aiogram import Bot

    if env_file.exists():
        raise RuntimeError(f"Configuration already exists: {env_file}")
    if timeout_seconds < 30 or timeout_seconds > 900:
        raise ValueError("Pairing timeout must be between 30 and 900 seconds")

    token = _read_token()
    code = secrets.token_hex(4).upper()
    bot = Bot(token)
    try:
        identity = await bot.get_me()
        await bot.delete_webhook(drop_pending_updates=True)
        command = f"/pair {code}"
        print(f"Open @{identity.username} and send this one-time command:")
        print(command)
        print(f"Waiting up to {timeout_seconds} seconds...")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        offset: int | None = None
        while loop.time() < deadline:
            remaining = max(1, int(deadline - loop.time()))
            updates = await bot.get_updates(
                offset=offset,
                timeout=min(10, remaining),
                allowed_updates=["message"],
            )
            for update in updates:
                offset = update.update_id + 1
                message = update.message
                if message is None or message.from_user is None or message.text is None:
                    continue
                if compare_digest(message.text.strip(), command):
                    return _write_configuration(
                        env_file, token, message.from_user.id
                    )
        raise TimeoutError("Pairing timed out; no configuration was written")
    finally:
        await bot.session.close()


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.command == "bot":
        asyncio.run(run_bot(settings))
        return
    if args.command == "configure-bot":
        configured = configure_bot(args.env_file)
        print(f"Configuration saved securely: {configured.resolve()}")
        return
    if args.command == "pair-bot":
        configured = asyncio.run(pair_bot(args.env_file, args.timeout))
        print(f"Configuration saved securely: {configured.resolve()}")
        return
    if args.command == "rotate-bot-token":
        configured = rotate_bot_token(
            args.env_file, token_from_stdin=args.token_stdin
        )
        print(f"Telegram token replaced securely: {configured.resolve()}")
        return
    if args.command == "configure-ocr":
        configured = configure_ocr(args.env_file, key_from_stdin=args.key_stdin)
        print(f"OCR configured securely: {configured.resolve()}")
        return
    if args.command == "demo-pdf":
        from autodkp.pdf.renderer import PdfRenderer

        renderer = PdfRenderer.from_defaults(settings)
        renderer.render(make_demo_deal(), args.output)
        print(args.output.resolve())


if __name__ == "__main__":
    main()

