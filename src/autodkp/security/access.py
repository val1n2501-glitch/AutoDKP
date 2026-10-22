from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject


def is_allowed_user(
    user_id: int | None,
    chat_type: str | None,
    allowed_ids: frozenset[int],
) -> bool:
    return user_id is not None and user_id in allowed_ids and chat_type == "private"


class AllowlistMiddleware(BaseMiddleware):
    def __init__(self, allowed_ids: frozenset[int]) -> None:
        self.allowed_ids = allowed_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        chat = data.get("event_chat")
        allowed = is_allowed_user(
            user.id if user is not None else None,
            str(chat.type) if chat is not None else None,
            self.allowed_ids,
        )
        if not allowed:
            if isinstance(event, Message):
                await event.answer("Доступ запрещён.")
            elif isinstance(event, CallbackQuery):
                await event.answer("Доступ запрещён.", show_alert=True)
            return None
        return await handler(event, data)

