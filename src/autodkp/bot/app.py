from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from contextlib import suppress
from datetime import timedelta
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    CopyTextButton,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from autodkp.config import Settings
from autodkp.domain.manual_entry import MANUAL_FIELDS, ManualEntryError
from autodkp.domain.manual_entry import FIELD_BY_KEY
from autodkp.domain.models import DealDraft, DocumentKind, Person
from autodkp.pdf.errors import FieldOverflowError, PdfRenderError
from autodkp.pdf.price_words import rubles_in_words
from autodkp.pdf.renderer import PdfRenderer
from autodkp.security.access import AllowlistMiddleware
from autodkp.storage.session_store import InMemoryDealStore
from autodkp.recognition.base import RecognitionProvider
from autodkp.recognition.image_security import (
    UnsafeImageError,
    assess_image_quality,
    sanitize_image,
)
from autodkp.recognition.openai_provider import OpenAIRecognitionProvider


logger = logging.getLogger(__name__)


class ManualDealStates(StatesGroup):
    entering = State()
    review = State()


class PhotoDealStates(StatesGroup):
    choosing_document = State()
    uploading = State()
    recognizing = State()
    reviewing = State()
    correcting = State()


_PDF_FIELD_LABELS = {
    "agreement_place": "Место составления",
    "buyer_details_line1": "Данные покупателя",
    "buyer_details_line2": "Паспорт покупателя",
    "seller_details_line1": "Данные продавца",
    "seller_details_line2": "Паспорт продавца",
    "vehicle_make_model": "Марка и модель",
    "vehicle_type": "Тип ТС",
    "vehicle_color": "Цвет кузова",
    "pts_issued_by": "Кем выдан ПТС",
    "sts_issued_by": "Кем выдано СТС",
}


def _start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📷 Заполнить по фото", callback_data="deal:photo")],
            [InlineKeyboardButton(text="✍️ Заполнить вручную", callback_data="deal:new")],
        ]
    )


def _document_keyboard(
    deal_id: str,
    revision: int,
    confirmed_documents: set[str] | None = None,
) -> InlineKeyboardMarkup:
    confirmed = confirmed_documents or set()

    def label(marker: str, text: str) -> str:
        return f"✅ 🔄 Заменить {text}" if marker in confirmed else text

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label("sts", "СТС автомобиля"), callback_data=f"photo-doc:{deal_id}:{revision}:sts")],
            [InlineKeyboardButton(text=label("paper_pts", "ПТС"), callback_data=f"photo-doc:{deal_id}:{revision}:pts")],
            [InlineKeyboardButton(text=label("passport:seller", "Паспорт продавца"), callback_data=f"photo-doc:{deal_id}:{revision}:seller")],
            [InlineKeyboardButton(text=label("passport:buyer", "Паспорт покупателя"), callback_data=f"photo-doc:{deal_id}:{revision}:buyer")],
            [InlineKeyboardButton(text="Дополнить вручную", callback_data=f"photo-manual:{deal_id}:{revision}")],
        ]
    )


def _deal_progress_text(session, *, missing_limit: int = 8) -> str:
    values = session.form.values if session.form is not None else {}
    missing = [spec.label for spec in MANUAL_FIELDS if spec.key not in values]
    confirmed = session.confirmed_documents
    documents = (
        ("paper_pts", "ПТС"),
        ("passport:seller", "Паспорт продавца"),
        ("passport:buyer", "Паспорт покупателя"),
        ("sts", "СТС (если данные не извлечены из ПТС)"),
    )
    document_lines = "\n".join(
        f"{'✅' if marker in confirmed else '▫️'} {label}"
        for marker, label in documents
    )
    rows = [
        f"Заполнено полей: {len(values)} из {len(MANUAL_FIELDS)}.",
        "",
        "Документы:",
        document_lines,
    ]
    if missing:
        shown = missing[:missing_limit]
        rows.extend(
            [
                "",
                f"PDF пока недоступен: осталось обязательных полей — {len(missing)}.",
                "Не заполнено: " + "; ".join(shown)
                + ("…" if len(missing) > len(shown) else "."),
            ]
        )
    else:
        rows.extend(["", "✅ Все обязательные поля заполнены — PDF доступен."])
    return "\n".join(rows)


def _recognized_keyboard(deal_id: str, revision: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Всё верно", callback_data=f"photo-confirm:{deal_id}:{revision}")],
            [InlineKeyboardButton(text="✏️ Изменить данные", callback_data=f"photo-edit:{deal_id}:{revision}")],
            [InlineKeyboardButton(text="📷 Загрузить фото заново", callback_data=f"photo-retry:{deal_id}:{revision}")],
            [InlineKeyboardButton(text="Не использовать, ввести вручную", callback_data=f"photo-manual:{deal_id}:{revision}")],
        ]
    )


def _recognized_sections_keyboard(
    deal_id: str, revision: int, field_keys: set[str]
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=label,
                callback_data=f"photo-section:{deal_id}:{revision}:{section}",
            )
        ]
        for section, label in _SECTIONS
        if any(_section_for(key) == section for key in field_keys)
    ]
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к проверке",
                callback_data=f"photo-review:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _recognized_fields_keyboard(
    section: str, deal_id: str, revision: int, field_keys: set[str]
) -> InlineKeyboardMarkup:
    rows = []
    for index, spec in enumerate(MANUAL_FIELDS):
        if spec.key in field_keys and _section_for(spec.key) == section:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=spec.label,
                        callback_data=f"photo-field:{deal_id}:{revision}:{index}",
                    )
                ]
            )
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к разделам",
                callback_data=f"photo-edit:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _recognized_all_fields_keyboard(
    deal_id: str,
    revision: int,
    field_keys: set[str],
    uncertain_keys: set[str] | None = None,
) -> InlineKeyboardMarkup:
    uncertain = uncertain_keys or set()
    rows = []
    for index, spec in enumerate(MANUAL_FIELDS):
        if spec.key not in field_keys:
            continue
        marker = "⚠️ " if spec.key in uncertain else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker}{spec.label}",
                    callback_data=f"photo-field:{deal_id}:{revision}:{index}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к проверке",
                callback_data=f"photo-review:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _pending_ocr_text(
    values: dict[str, str],
    required_corrections: set[str] | None = None,
    uncertain_fields: set[str] | None = None,
) -> str:
    rows = ["Проверьте исправленные данные:\n"]
    required = required_corrections or set()
    uncertain = (uncertain_fields or set()) | required
    for spec in MANUAL_FIELDS:
        if spec.key in values:
            marker = "⚠️ " if spec.key in uncertain else ""
            rows.append(f"{marker}{spec.label}: {_text_preview(values[spec.key])}")
    if required:
        rows.append("\n⚠️ Отмеченные поля нужно исправить перед подтверждением.")
    elif uncertain:
        rows.append("\n⚠️ Отмеченные поля распознаны неуверенно — проверьте их внимательно.")
    return "\n".join(rows)


def _text_preview(value: str, limit: int = 160) -> str:
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


_DOCUMENT_TITLES = {
    DocumentKind.STS.value: "СТС автомобиля",
    DocumentKind.PAPER_PTS.value: "Бумажный ПТС",
    DocumentKind.PASSPORT.value: "Паспорт",
}


def _specific_document_title(
    document_kind: str | None, passport_role: str | None = None
) -> str:
    if document_kind == DocumentKind.PASSPORT.value:
        if passport_role == "seller":
            return "Паспорт продавца"
        if passport_role == "buyer":
            return "Паспорт покупателя"
    return _DOCUMENT_TITLES.get(document_kind or "", "Документ")


def _upload_summary_text(
    document_kind: str,
    count: int,
    *,
    passport_role: str | None = None,
    replace_index: int | None = None,
) -> str:
    title = _DOCUMENT_TITLES.get(document_kind, "Документ")
    if document_kind == DocumentKind.PASSPORT.value:
        title += " продавца" if passport_role == "seller" else " покупателя"
    rows = [f"{title}\n", f"Загружено фото: {count} из 4."]
    rows.extend(f"{index}. Фото {index} — готово" for index in range(1, count + 1))
    if replace_index is not None:
        rows.append(
            f"\nРежим замены: следующий снимок заменит фото {replace_index + 1}."
        )
    elif count == 0:
        rows.append("\nОтправьте фото как изображение или JPG/PNG-файл.")
    else:
        rows.append("\nМожно добавить ещё фото или запустить распознавание.")
    return "\n".join(rows)


def _upload_keyboard(
    deal_id: str,
    revision: int,
    count: int,
    *,
    batch_version: int = 0,
    replace_index: int | None = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for index in range(count):
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🔄 Заменить фото {index + 1}",
                    callback_data=f"pr:{deal_id}:{revision}:{batch_version}:{index}",
                ),
                InlineKeyboardButton(
                    text=f"🗑 Удалить {index + 1}",
                    callback_data=f"pd:{deal_id}:{revision}:{batch_version}:{index}",
                ),
            ]
        )
    if replace_index is not None:
        rows.append(
            [
                InlineKeyboardButton(
                    text="Отменить замену",
                    callback_data=f"pc:{deal_id}:{revision}:{batch_version}",
                )
            ]
        )
    if count:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🔎 Распознать {count} фото",
                    callback_data=f"pg:{deal_id}:{revision}:{batch_version}",
                )
            ]
        )
    rows.extend(
        [
            [InlineKeyboardButton(text="Выбрать другой документ", callback_data=f"photo-retry:{deal_id}:{revision}")],
            [InlineKeyboardButton(text="Дополнить вручную", callback_data=f"photo-manual:{deal_id}:{revision}")],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


class _LimitedBuffer(io.BytesIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > self._limit:
            raise UnsafeImageError("Файл слишком большой.")
        return super().write(data)


_SECTIONS: tuple[tuple[str, str], ...] = (
    ("deal", "Договор"),
    ("seller", "Продавец"),
    ("buyer", "Покупатель"),
    ("vehicle", "Автомобиль"),
    ("pts", "ПТС"),
    ("sts", "СТС"),
    ("price", "Цена"),
)


def _section_for(key: str) -> str:
    if key.startswith("seller."):
        return "seller"
    if key.startswith("buyer."):
        return "buyer"
    if key.startswith("vehicle.pts."):
        return "pts"
    if key.startswith("vehicle.sts."):
        return "sts"
    if key.startswith("vehicle."):
        return "vehicle"
    if key == "price_rubles":
        return "price"
    return "deal"


def _review_keyboard(deal_id: str, revision: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Сформировать PDF",
                    callback_data=f"review-pdf:{deal_id}:{revision}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔎 Проверить все данные",
                    callback_data=f"review-details:{deal_id}:{revision}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Изменить данные",
                    callback_data=f"review-edit:{deal_id}:{revision}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Завершить и удалить данные",
                    callback_data=f"review-finish:{deal_id}:{revision}",
                )
            ],
        ]
    )


def _after_pdf_keyboard(deal_id: str, revision: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Изменить данные",
                    callback_data=f"review-edit:{deal_id}:{revision}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Начать новую сделку",
                    callback_data=f"review-new:{deal_id}:{revision}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Завершить и удалить данные",
                    callback_data=f"review-finish:{deal_id}:{revision}",
                )
            ],
        ]
    )


def _section_keyboard(deal_id: str, revision: int) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=label,
                callback_data=f"edit-section:{deal_id}:{revision}:{key}",
            )
        ]
        for key, label in _SECTIONS
    ]
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад", callback_data=f"edit-back:{deal_id}:{revision}"
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _field_keyboard(
    section: str, deal_id: str, revision: int
) -> InlineKeyboardMarkup:
    rows = []
    for index, spec in enumerate(MANUAL_FIELDS):
        if _section_for(spec.key) == section:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=spec.label,
                        callback_data=f"edit-field:{deal_id}:{revision}:{index}",
                    )
                ]
            )
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к разделам",
                callback_data=f"review-edit:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _manual_missing_sections_keyboard(
    deal_id: str, revision: int, missing_keys: set[str]
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=label,
                callback_data=f"manual-section:{deal_id}:{revision}:{key}",
            )
        ]
        for key, label in _SECTIONS
        if any(_section_for(field_key) == key for field_key in missing_keys)
    ]
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к документам",
                callback_data=f"manual-return:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _manual_missing_fields_keyboard(
    section: str,
    deal_id: str,
    revision: int,
    missing_keys: set[str],
) -> InlineKeyboardMarkup:
    rows = []
    for index, spec in enumerate(MANUAL_FIELDS):
        if spec.key in missing_keys and _section_for(spec.key) == section:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=spec.label,
                        callback_data=f"manual-field:{deal_id}:{revision}:{index}",
                    )
                ]
            )
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к разделам",
                callback_data=f"photo-manual:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _manual_missing_all_fields_keyboard(
    deal_id: str, revision: int, missing_keys: set[str]
) -> InlineKeyboardMarkup:
    rows = []
    for index, spec in enumerate(MANUAL_FIELDS):
        if spec.key in missing_keys:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=spec.label,
                        callback_data=f"manual-field:{deal_id}:{revision}:{index}",
                    )
                ]
            )
    rows.append(
        [
            InlineKeyboardButton(
                text="Назад к документам",
                callback_data=f"manual-return:{deal_id}:{revision}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _callback_parts(data: str | None, prefix: str, count: int) -> tuple[str, ...] | None:
    if data is None:
        return None
    parts = data.split(":")
    if len(parts) != count + 1 or parts[0] != prefix:
        return None
    return tuple(parts[1:])


def _masked_passport(person: Person) -> str:
    series = f"{person.passport_series[:2]} ••"
    number = f"••••{person.passport_number[-2:]}"
    return f"{series} {number}"


def _review_text(deal: DealDraft) -> str:
    price = f"{deal.price_rubles:,}".replace(",", " ")
    return (
        "Проверьте основные данные перед формированием PDF.\n\n"
        f"Договор: {deal.agreement_place}, {deal.agreement_date:%d.%m.%Y}\n"
        f"Продавец: {deal.seller.full_name} · {_masked_passport(deal.seller)}\n"
        f"Покупатель: {deal.buyer.full_name} · {_masked_passport(deal.buyer)}\n"
        f"Автомобиль: {deal.vehicle.make_model}\n"
        f"VIN: {deal.vehicle.vin}\n"
        f"Госномер: {deal.vehicle.registration_plate}\n"
        f"Стоимость: {price} ₽\n\n"
        "Подписи в PDF останутся пустыми для заполнения от руки."
    )


def _person_details(title: str, person: Person) -> str:
    return (
        f"{title}\n"
        f"ФИО: {person.full_name}\n"
        f"Дата рождения: {person.birth_date:%d.%m.%Y}\n"
        f"Место рождения: {person.birth_place}\n"
        f"Адрес: {person.address}\n"
        f"Паспорт: {person.passport_series} № {person.passport_number}\n"
        f"Кем выдан: {person.passport_issued_by}\n"
        f"Дата выдачи: {person.passport_issue_date:%d.%m.%Y}"
    )


def _detailed_review_texts(deal: DealDraft) -> tuple[str, ...]:
    vehicle = deal.vehicle
    price = f"{deal.price_rubles:,}".replace(",", " ")
    return (
        (
            "ДОГОВОР\n"
            f"Место составления: {deal.agreement_place}\n"
            f"Дата: {deal.agreement_date:%d.%m.%Y}"
        ),
        _person_details("ПРОДАВЕЦ", deal.seller),
        _person_details("ПОКУПАТЕЛЬ", deal.buyer),
        (
            "АВТОМОБИЛЬ\n"
            f"Марка и модель: {vehicle.make_model}\n"
            f"VIN: {vehicle.vin}\n"
            f"Тип: {vehicle.vehicle_type}\n"
            f"Год выпуска: {vehicle.manufacture_year}\n"
            f"Пробег: {vehicle.mileage_km} км\n"
            f"Мощность: {vehicle.engine_power_hp} л. с.\n"
            f"Рабочий объём: {vehicle.engine_displacement_cc} куб. см\n"
            f"Цвет: {vehicle.body_color}\n"
            f"Модель двигателя: {vehicle.engine_model}\n"
            f"Номер двигателя: {vehicle.engine_number}\n"
            f"Номер шасси/рамы: {vehicle.chassis_number}\n"
            f"Номер кузова: {vehicle.body_number}\n"
            f"Госномер: {vehicle.registration_plate}"
        ),
        (
            "ПТС\n"
            f"Серия и номер: {vehicle.pts.series_number}\n"
            f"Кем выдан: {vehicle.pts.issued_by}\n"
            f"Дата выдачи: {vehicle.pts.issue_date:%d.%m.%Y}"
        ),
        (
            "СТС\n"
            f"Серия и номер: {vehicle.sts.series_number}\n"
            f"Кем выдано: {vehicle.sts.issued_by}\n"
            f"Дата выдачи: {vehicle.sts.issue_date:%d.%m.%Y}"
        ),
        (
            "СТОИМОСТЬ\n"
            f"Цифрами: {price} ₽\n"
            f"Прописью: {rubles_in_words(deal.price_rubles)}"
        ),
    )


def build_dispatcher(
    settings: Settings,
    store: InMemoryDealStore | None = None,
    recognition_provider: RecognitionProvider | None = None,
) -> Dispatcher:
    dispatcher = Dispatcher()
    deal_store = store or InMemoryDealStore(timedelta(minutes=settings.deal_ttl_minutes))
    dispatcher["deal_store"] = deal_store
    middleware = AllowlistMiddleware(settings.allowed_telegram_ids)
    dispatcher.message.outer_middleware(middleware)
    dispatcher.callback_query.outer_middleware(middleware)

    router = Router(name="core")
    photo_batches: dict[int, list[bytes]] = {}
    photo_batch_versions: dict[int, int] = {}
    photo_batch_locks: dict[int, asyncio.Lock] = {}
    photo_ui_locks: dict[int, asyncio.Lock] = {}
    photo_replace_reservations: set[tuple[int, int]] = set()
    photo_recognizing: set[tuple[int, str, int]] = set()
    photo_ocr_tasks: dict[tuple[int, str], asyncio.Task] = {}

    def cleanup_photo_state(user_id: int, deal_id: str) -> None:
        photo_batches.pop(user_id, None)
        photo_batch_versions.pop(user_id, None)
        photo_ui_locks.pop(user_id, None)
        photo_replace_reservations.difference_update(
            reservation
            for reservation in photo_replace_reservations
            if reservation[0] == user_id
        )
        for marker in tuple(photo_recognizing):
            if marker[:2] == (user_id, deal_id):
                photo_recognizing.discard(marker)
        task = photo_ocr_tasks.pop((user_id, deal_id), None)
        if task is not None and not task.done():
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            if task is not current:
                task.cancel()

    deal_store.add_delete_callback(cleanup_photo_state)

    if recognition_provider is None and settings.ocr_provider == "openai":
        if settings.openai_api_key is not None:
            recognition_provider = OpenAIRecognitionProvider(
                settings.openai_api_key.get_secret_value(), settings.openai_ocr_model
            )

    async def render_upload_panel(
        user_id: int,
        state: FSMContext,
        bot: Bot,
        fallback_message: Message,
        *,
        move_after_input: bool = False,
    ) -> None:
        """Keep one current bot panel while the user sends several photos."""
        ui_lock = photo_ui_locks.setdefault(user_id, asyncio.Lock())
        async with ui_lock:
            data = await state.get_data()
            deal_id = data.get("deal_id")
            revision = data.get("revision")
            document_kind = data.get("document_kind")
            passport_role = data.get("passport_role")
            replace_index = data.get("replace_photo_index")
            if (
                not isinstance(deal_id, str)
                or not isinstance(revision, int)
                or not isinstance(document_kind, str)
            ):
                return
            lock = photo_batch_locks.setdefault(user_id, asyncio.Lock())
            async with lock:
                count = len(photo_batches.get(user_id, []))
                batch_version = photo_batch_versions.get(user_id, 0)
            text = _upload_summary_text(
                document_kind,
                count,
                passport_role=passport_role if isinstance(passport_role, str) else None,
                replace_index=replace_index if isinstance(replace_index, int) else None,
            )
            session = deal_store.get(user_id)
            if session is not None and session.deal_id == deal_id:
                text = _deal_progress_text(session) + "\n\n" + text
            keyboard = _upload_keyboard(
                deal_id,
                revision,
                count,
                batch_version=batch_version,
                replace_index=replace_index if isinstance(replace_index, int) else None,
            )
            chat_id = data.get("upload_panel_chat_id")
            message_id = data.get("upload_panel_message_id")
            if move_after_input:
                panel = await fallback_message.answer(text, reply_markup=keyboard)
                await state.update_data(
                    upload_panel_chat_id=panel.chat.id,
                    upload_panel_message_id=panel.message_id,
                )
                if (
                    isinstance(chat_id, int)
                    and isinstance(message_id, int)
                    and (chat_id, message_id) != (panel.chat.id, panel.message_id)
                ):
                    try:
                        await bot.delete_message(chat_id=chat_id, message_id=message_id)
                    except TelegramBadRequest:
                        pass
                return
            if isinstance(chat_id, int) and isinstance(message_id, int):
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        reply_markup=keyboard,
                    )
                    return
                except TelegramBadRequest as error:
                    if "message is not modified" in str(error).lower():
                        return
            panel = await fallback_message.answer(text, reply_markup=keyboard)
            await state.update_data(
                upload_panel_chat_id=panel.chat.id,
                upload_panel_message_id=panel.message_id,
            )

    async def begin_manual_entry(
        message: Message,
        state: FSMContext,
        session,
        *,
        intro: str,
    ) -> None:
        assert session.form is not None
        first = session.form.next_missing()
        if first is None:
            deal = await deal_store.finalize_manual_form(
                session.owner_id, session.deal_id
            )
            current = deal_store.get(session.owner_id)
            assert current is not None
            await state.set_state(ManualDealStates.review)
            await state.set_data(
                {"deal_id": session.deal_id, "revision": current.revision}
            )
            await message.answer(
                _review_text(deal),
                reply_markup=_review_keyboard(session.deal_id, current.revision),
            )
            return
        await state.set_state(ManualDealStates.entering)
        await state.set_data(
            {
                "deal_id": session.deal_id,
                "revision": session.revision,
                "field_key": first.key,
                "edit_mode": False,
            }
        )
        await message.answer(f"{intro}\n\n{first.prompt}")

    @router.message(CommandStart())
    async def start(
        message: Message,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        reset_note = ""
        if message.from_user is not None:
            previous = deal_store.get(message.from_user.id)
            if previous is not None:
                await deal_store.delete_if_current(
                    message.from_user.id, previous.deal_id
                )
                reset_note = "Предыдущий незавершённый черновик удалён.\n\n"
        await state.clear()
        await message.answer(
            reset_note
            + "AutoDKP готов к работе. Данные черновика хранятся только временно "
            "в памяти программы и удаляются при отмене или истечении срока. "
            "Отправленные сообщения при этом могут сохраняться в Telegram.",
            reply_markup=_start_keyboard(),
        )

    @router.message(Command("help"))
    async def show_help(message: Message) -> None:
        await message.answer(
            "Как подготовить ДКП:\n\n"
            "1. Нажмите «Заполнить по фото».\n"
            "2. Загрузите ПТС, паспорт продавца и паспорт покупателя. "
            "СТС можно добавить отдельно, если его данные не извлечены из ПТС.\n"
            "3. Проверьте распознанные значения и исправьте поля с ⚠️.\n"
            "4. Через «Дополнить вручную» внесите место и дату договора, "
            "пробег и стоимость.\n"
            "5. Проверьте все данные и сформируйте PDF.\n\n"
            "Команды: /status — текущий прогресс, /cancel — удалить черновик, "
            "/start — начать заново."
        )

    @router.message(Command("status"))
    async def show_status(message: Message, deal_store: InMemoryDealStore) -> None:
        if message.from_user is None:
            return
        session = deal_store.get(message.from_user.id)
        if session is None:
            await message.answer(
                "Активной сделки нет. Начните через /start.",
                reply_markup=_start_keyboard(),
            )
            return
        if session.draft is not None:
            await message.answer(
                "✅ Все обязательные поля заполнены. PDF доступен.\n\n"
                + _review_text(session.draft),
                reply_markup=_review_keyboard(session.deal_id, session.revision),
            )
            return
        await message.answer(
            _deal_progress_text(session),
            reply_markup=_document_keyboard(
                session.deal_id,
                session.revision,
                session.confirmed_documents,
            ),
        )

    @router.callback_query(F.data == "deal:new")
    async def create_manual_deal(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        if callback.from_user is None:
            return
        if not isinstance(callback.message, Message):
            await callback.answer("Сообщение больше недоступно", show_alert=True)
            return
        if deal_store.get(callback.from_user.id) is not None:
            await callback.answer(
                "У вас уже есть активная сделка. Сначала завершите её или используйте /cancel.",
                show_alert=True,
            )
            return
        session = deal_store.create(callback.from_user.id)
        await callback.answer()
        await begin_manual_entry(
            callback.message,
            state,
            session,
            intro="Начинаем ручное заполнение. Для отмены используйте /cancel.",
        )

    @router.callback_query(F.data == "deal:photo")
    async def start_photo_deal(callback: CallbackQuery) -> None:
        if not isinstance(callback.message, Message):
            await callback.answer("Сообщение больше недоступно", show_alert=True)
            return
        if recognition_provider is None:
            await callback.answer()
            await callback.message.answer(
                "Функция фото уже добавлена, но облачное распознавание пока не настроено. "
                "Фото никуда не отправлено. Можно продолжить вручную.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="Заполнить вручную", callback_data="deal:new")]
                    ]
                ),
            )
            return
        await callback.answer()
        await callback.message.answer(
            "Фото документа будет очищено от EXIF/GPS и передано OpenAI только для "
            "извлечения полей ДКП. Запрос отправляется с параметром store=False; "
            "сроки возможного технического хранения определяются политикой провайдера. "
            "Исходное сообщение и файл могут сохраняться в Telegram.\n\n"
            "Продолжая, вы подтверждаете, что имеете право обрабатывать данные людей в этих документах.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="Понятно, продолжить", callback_data="photo-consent")],
                    [InlineKeyboardButton(text="Нет, заполнить вручную", callback_data="deal:new")],
                ]
            ),
        )

    @router.callback_query(F.data == "photo-consent")
    async def accept_photo_consent(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        if not isinstance(callback.message, Message):
            return
        if deal_store.get(callback.from_user.id) is not None:
            await callback.answer("Уже есть активная сделка", show_alert=True)
            return
        session = deal_store.create(callback.from_user.id)
        await state.set_state(PhotoDealStates.choosing_document)
        await state.set_data({"deal_id": session.deal_id, "revision": session.revision})
        await callback.answer()
        await callback.message.answer(
            "Выберите, какой документ добавим. Можно начать с любого.",
            reply_markup=_document_keyboard(session.deal_id, session.revision),
        )

    @router.callback_query(
        PhotoDealStates.choosing_document, F.data.startswith("photo-doc:")
    )
    async def choose_photo_document(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        if not isinstance(callback.message, Message):
            return
        parts = _callback_parts(callback.data, "photo-doc", 3)
        if parts is None:
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        callback_deal_id, raw_revision, choice = parts
        session = deal_store.get(callback.from_user.id)
        data = await state.get_data()
        if (
            session is None
            or session.deal_id != callback_deal_id
            or str(session.revision) != raw_revision
            or data.get("deal_id") != callback_deal_id
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        mapping = {
            "sts": (DocumentKind.STS, None, "СТС: пришлите лицевую и оборотную стороны."),
            "pts": (DocumentKind.PAPER_PTS, None, "Бумажный ПТС: пришлите фото всех нужных разворотов."),
            "seller": (DocumentKind.PASSPORT, "seller", "Паспорт продавца: пришлите основной разворот и страницу регистрации."),
            "buyer": (DocumentKind.PASSPORT, "buyer", "Паспорт покупателя: пришлите основной разворот и страницу регистрации."),
        }
        selected = mapping.get(choice)
        if selected is None:
            await callback.answer("Неизвестный документ", show_alert=True)
            return
        await callback.answer()
        kind, role, prompt = selected
        photo_batches[callback.from_user.id] = []
        photo_batch_versions[callback.from_user.id] = 0
        await state.set_state(PhotoDealStates.uploading)
        await state.update_data(
            document_kind=kind.value,
            passport_role=role,
            replace_photo_index=None,
            replace_batch_version=None,
        )
        current = await state.get_data()
        deal_id = current.get("deal_id")
        revision = current.get("revision")
        if not isinstance(deal_id, str) or not isinstance(revision, int):
            await state.clear()
            await callback.message.answer("Операция устарела. Начните заново через /start.")
            return
        await callback.message.edit_text(
            _deal_progress_text(session)
            + "\n\n"
            + prompt
            + "\n\n"
            + _upload_summary_text(kind.value, 0, passport_role=role),
            reply_markup=_upload_keyboard(deal_id, revision, 0, batch_version=0),
        )
        await state.update_data(
            upload_panel_chat_id=callback.message.chat.id,
            upload_panel_message_id=callback.message.message_id,
        )

    @router.message(PhotoDealStates.recognizing, F.photo | F.document)
    async def reject_photo_while_recognizing(message: Message) -> None:
        await message.answer(
            "Сейчас распознаю уже загруженные фото. Дождитесь результата, "
            "после этого можно будет выбрать следующий документ."
        )

    @router.message(PhotoDealStates.uploading, F.photo | F.document)
    async def collect_photo(
        message: Message, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        if message.from_user is None or recognition_provider is None:
            return
        state_data = await state.get_data()
        deal_id = state_data.get("deal_id")
        revision = state_data.get("revision")
        kind_raw = state_data.get("document_kind")
        session = deal_store.get(message.from_user.id)
        if (
            not isinstance(deal_id, str)
            or not isinstance(revision, int)
            or not isinstance(kind_raw, str)
            or session is None
            or session.deal_id != deal_id
            or session.revision != revision
        ):
            await state.clear()
            await message.answer("Операция устарела. Начните заново через /start.")
            return
        media = message.photo[-1] if message.photo else message.document
        if media is None:
            return
        if media.file_size is not None and media.file_size > settings.ocr_max_file_bytes:
            await message.answer("Файл слишком большой. Максимум — 10 МБ.")
            return
        if message.document is not None and message.document.mime_type not in {"image/jpeg", "image/png"}:
            await message.answer("Пока поддерживаются только JPG и PNG.")
            return
        lock = photo_batch_locks.setdefault(message.from_user.id, asyncio.Lock())
        replacement_reservation: tuple[int, int] | None = None
        async with lock:
            current_batch = photo_batches.get(message.from_user.id, [])
            replace_index = state_data.get("replace_photo_index")
            replace_batch_version = state_data.get("replace_batch_version")
            replacing_existing = (
                isinstance(replace_index, int)
                and 0 <= replace_index < len(current_batch)
                and isinstance(replace_batch_version, int)
                and photo_batch_versions.get(message.from_user.id, 0)
                == replace_batch_version
            )
            if isinstance(replace_index, int) and not replacing_existing:
                await state.update_data(
                    replace_photo_index=None,
                    replace_batch_version=None,
                )
                await message.answer(
                    "Список фотографий уже изменился. Нажмите «Заменить» ещё раз."
                )
                return
            if len(current_batch) >= 4 and not replacing_existing:
                await message.answer(
                    "Уже загружено 4 фото. Удалите или замените один из снимков."
                )
                return
            if replacing_existing:
                replacement_reservation = (message.from_user.id, replace_index)
                if replacement_reservation in photo_replace_reservations:
                    await message.answer(
                        "Замена этого фото уже загружается. Дождитесь обновления списка."
                    )
                    return
                photo_replace_reservations.add(replacement_reservation)
        buffer = _LimitedBuffer(settings.ocr_max_file_bytes)
        try:
            try:
                await message.bot.download(media.file_id, destination=buffer)
                sanitized = await asyncio.wait_for(
                    asyncio.to_thread(
                        sanitize_image,
                        buffer.getvalue(),
                        max_bytes=settings.ocr_max_file_bytes,
                        max_pixels=settings.ocr_max_image_pixels,
                    ),
                    timeout=15,
                )
                quality_warnings = await asyncio.to_thread(
                    assess_image_quality, sanitized
                )
            except UnsafeImageError as error:
                await message.answer(str(error))
                return
            except (TimeoutError, asyncio.TimeoutError):
                await message.answer(
                    "Не успел загрузить и проверить фото. Отправьте его ещё раз."
                )
                return
            except Exception:
                await message.answer(
                    "Не удалось загрузить фото. Уже добавленные снимки сохранены; попробуйте ещё раз."
                )
                return
            finally:
                buffer.close()

            async with lock:
                batch = photo_batches.setdefault(message.from_user.id, [])
                replace_index = state_data.get("replace_photo_index")
                if isinstance(replace_index, int) and 0 <= replace_index < len(batch):
                    if photo_batch_versions.get(message.from_user.id, 0) != state_data.get(
                        "replace_batch_version"
                    ):
                        await state.update_data(
                            replace_photo_index=None,
                            replace_batch_version=None,
                        )
                        await message.answer(
                            "Список фотографий уже изменился. Нажмите «Заменить» ещё раз."
                        )
                        return
                    batch[replace_index] = sanitized
                    photo_batch_versions[message.from_user.id] = (
                        photo_batch_versions.get(message.from_user.id, 0) + 1
                    )
                elif len(batch) >= 4:
                    await message.answer("Для одного документа можно добавить не более 4 фото.")
                    return
                else:
                    batch.append(sanitized)
                    photo_batch_versions[message.from_user.id] = (
                        photo_batch_versions.get(message.from_user.id, 0) + 1
                    )
        finally:
            if replacement_reservation is not None:
                async with lock:
                    photo_replace_reservations.discard(replacement_reservation)
        await state.update_data(
            replace_photo_index=None,
            replace_batch_version=None,
        )
        if quality_warnings:
            await message.answer(
                "⚠️ Проверьте качество снимка: "
                + "; ".join(quality_warnings)
                + ". Фото сохранено, но лучше заменить его до распознавания, "
                "если текст читается плохо."
            )
        await render_upload_panel(
            message.from_user.id,
            state,
            message.bot,
            message,
            move_after_input=True,
        )

    @router.callback_query(
        PhotoDealStates.uploading, F.data.startswith("pr:")
    )
    async def replace_uploaded_photo(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "pr", 4)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, raw_batch_version, raw_index = parts
        data = await state.get_data()
        session = deal_store.get(callback.from_user.id)
        try:
            revision = int(raw_revision)
            batch_version = int(raw_batch_version)
            index = int(raw_index)
        except ValueError:
            await callback.answer("Неизвестное фото", show_alert=True)
            return
        batch = photo_batches.get(callback.from_user.id, [])
        if (
            session is None
            or session.deal_id != deal_id
            or session.revision != revision
            or data.get("deal_id") != deal_id
            or photo_batch_versions.get(callback.from_user.id, 0) != batch_version
            or not 0 <= index < len(batch)
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.update_data(
            replace_photo_index=index,
            replace_batch_version=batch_version,
        )
        await callback.answer(f"Следующее фото заменит фото {index + 1}")
        await render_upload_panel(
            callback.from_user.id, state, callback.bot, callback.message
        )

    @router.callback_query(
        PhotoDealStates.uploading, F.data.startswith("pc:")
    )
    async def cancel_uploaded_photo_replacement(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "pc", 3)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, raw_batch_version = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or str(photo_batch_versions.get(callback.from_user.id, 0))
            != raw_batch_version
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.update_data(
            replace_photo_index=None,
            replace_batch_version=None,
        )
        await callback.answer("Замена отменена")
        await render_upload_panel(
            callback.from_user.id, state, callback.bot, callback.message
        )

    @router.callback_query(
        PhotoDealStates.uploading, F.data.startswith("pd:")
    )
    async def remove_uploaded_photo(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "pd", 4)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, raw_batch_version, raw_index = parts
        session = deal_store.get(callback.from_user.id)
        try:
            revision = int(raw_revision)
            batch_version = int(raw_batch_version)
            index = int(raw_index)
        except ValueError:
            await callback.answer("Неизвестное фото", show_alert=True)
            return
        lock = photo_batch_locks.setdefault(callback.from_user.id, asyncio.Lock())
        async with lock:
            batch = photo_batches.get(callback.from_user.id, [])
            if (
                session is None
                or session.deal_id != deal_id
                or session.revision != revision
                or photo_batch_versions.get(callback.from_user.id, 0)
                != batch_version
                or not 0 <= index < len(batch)
            ):
                await callback.answer("Эта кнопка устарела", show_alert=True)
                return
            batch.pop(index)
            photo_batch_versions[callback.from_user.id] = batch_version + 1
        await state.update_data(replace_photo_index=None)
        await callback.answer(f"Фото {index + 1} удалено")
        await render_upload_panel(
            callback.from_user.id, state, callback.bot, callback.message
        )

    @router.callback_query(PhotoDealStates.uploading, F.data.startswith("pg:"))
    async def recognize_photo_batch(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        if not isinstance(callback.message, Message) or recognition_provider is None:
            return
        parts = _callback_parts(callback.data, "pg", 3)
        if parts is None:
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        callback_deal_id, callback_revision, callback_batch_version = parts
        state_data = await state.get_data()
        deal_id = state_data.get("deal_id")
        revision = state_data.get("revision")
        kind_raw = state_data.get("document_kind")
        session = deal_store.get(callback.from_user.id)
        if (
            not isinstance(deal_id, str)
            or not isinstance(revision, int)
            or not isinstance(kind_raw, str)
            or session is None
            or session.deal_id != deal_id
            or session.revision != revision
            or callback_deal_id != deal_id
            or callback_revision != str(revision)
            or callback_batch_version
            != str(photo_batch_versions.get(callback.from_user.id, 0))
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        lock = photo_batch_locks.setdefault(callback.from_user.id, asyncio.Lock())
        recognition_marker = (callback.from_user.id, deal_id, revision)
        async with lock:
            if recognition_marker in photo_recognizing:
                await callback.answer("Распознавание уже выполняется", show_alert=True)
                return
            images = photo_batches.pop(callback.from_user.id, [])
            if images:
                photo_recognizing.add(recognition_marker)
        if not images:
            await callback.answer("Сначала отправьте хотя бы одно фото", show_alert=True)
            return
        task_key = (callback.from_user.id, deal_id)
        current_task = asyncio.current_task()
        if current_task is not None:
            photo_ocr_tasks[task_key] = current_task
        try:
            await state.set_state(PhotoDealStates.recognizing)
            await callback.answer("Распознаю…")
            await callback.message.edit_text(
                f"🔎 Распознаю документ\n\n"
                f"Обрабатываю {len(images)} фото. Обычно это занимает до минуты…"
            )
            result = await recognition_provider.recognize(
                DocumentKind(kind_raw),
                images,
                passport_role=state_data.get("passport_role"),
            )
        except asyncio.CancelledError:
            return
        except (TimeoutError, asyncio.TimeoutError):
            current = deal_store.get(callback.from_user.id)
            if (
                current is None
                or current.deal_id != deal_id
                or current.revision != revision
            ):
                return
            await state.set_state(PhotoDealStates.choosing_document)
            await callback.message.edit_text(
                "Распознавание заняло слишком много времени. "
                "Выберите документ и перешлите фото или продолжите вручную.",
                reply_markup=_document_keyboard(deal_id, revision),
            )
            return
        except Exception:
            current = deal_store.get(callback.from_user.id)
            if (
                current is None
                or current.deal_id != deal_id
                or current.revision != revision
            ):
                return
            await state.set_state(PhotoDealStates.choosing_document)
            await callback.message.edit_text(
                "Не удалось распознать документ. Фото удалены из оперативной памяти. "
                "Выерите документ для повтора или продолжите вручную.",
                reply_markup=_document_keyboard(deal_id, revision),
            )
            return
        finally:
            photo_recognizing.discard(recognition_marker)
            if photo_ocr_tasks.get(task_key) is current_task:
                photo_ocr_tasks.pop(task_key, None)

        current = deal_store.get(callback.from_user.id)
        if (
            current is None
            or current.deal_id != deal_id
            or current.revision != revision
        ):
            return

        expected_kind = DocumentKind(kind_raw)
        if result.document_kind != expected_kind:
            names = {
                DocumentKind.STS: "СТС",
                DocumentKind.PAPER_PTS: "бумажный ПТС",
                DocumentKind.PASSPORT: "паспорт",
                DocumentKind.EPTS_EXTRACT: "выписка ЭПТС",
                DocumentKind.UNKNOWN: "другой документ",
            }
            actual = names.get(result.document_kind, "другой документ")
            expected = names.get(expected_kind, "выбранный тип")
            await state.set_state(PhotoDealStates.choosing_document)
            await callback.message.edit_text(
                f"На фото похоже {actual}, а выбран {expected}. "
                "Выберите правильный тип и отправьте снимки снова.",
                reply_markup=_document_keyboard(deal_id, revision),
            )
            return

        accepted: dict[str, str] = {}
        required_corrections: set[str] = set()
        uncertain_fields: set[str] = set()
        alternatives: dict[str, tuple[str, ...]] = {}
        lines: list[str] = []
        skipped = 0
        skipped_keys: set[str] = set()
        grouped: dict[str, list] = {}
        for item in result.fields:
            grouped.setdefault(item.key, []).append(item)
        for key, candidates in grouped.items():
            spec = FIELD_BY_KEY[key]
            distinct = {" ".join(item.value.lower().split()) for item in candidates}
            if len(distinct) > 1:
                rendered = " / ".join(_text_preview(item.value) for item in candidates)
                preferred = max(candidates, key=lambda candidate: candidate.confidence)
                try:
                    spec.parser(preferred.value)
                except ManualEntryError:
                    required_corrections.add(key)
                accepted[key] = preferred.value
                required_corrections.add(key)
                uncertain_fields.add(key)
                alternatives[key] = tuple(item.value for item in candidates)
                lines.append(
                    f"⚠️ {spec.label}: {rendered} — найдено несколько записей, "
                    "выберите «Изменить данные»"
                )
                continue
            item = max(candidates, key=lambda candidate: candidate.confidence)
            try:
                spec.parser(item.value)
            except ManualEntryError:
                accepted[key] = item.value
                required_corrections.add(key)
                uncertain_fields.add(key)
                lines.append(
                    f"⚠️ {spec.label}: {_text_preview(item.value)} — "
                    "распознано в неверном формате, исправьте значение"
                )
                continue
            accepted[key] = item.value
            marker = "⚠️ " if item.confidence < 0.8 or item.warning else ""
            if marker:
                uncertain_fields.add(key)
            suffix = " — проверьте внимательно" if marker else ""
            lines.append(
                f"{marker}{spec.label}: {_text_preview(item.value)}{suffix}"
            )
        if not accepted:
            detail = "\n".join(lines)
            await state.set_state(PhotoDealStates.choosing_document)
            await callback.message.edit_text(
                "На фото не нашлось данных, которые можно безопасно подставить."
                + (f"\n\n{detail}" if detail else ""),
                reply_markup=_document_keyboard(deal_id, revision),
            )
            return
        passport_role = state_data.get("passport_role")
        if expected_kind is DocumentKind.PASSPORT and passport_role in {
            "seller",
            "buyer",
        }:
            current_session = deal_store.get(callback.from_user.id)
            other_role = "buyer" if passport_role == "seller" else "seller"
            selected_series = "".join(
                char
                for char in accepted.get(f"{passport_role}.passport_series", "")
                if char.isdigit()
            )
            selected_number = "".join(
                char
                for char in accepted.get(f"{passport_role}.passport_number", "")
                if char.isdigit()
            )
            other_values = (
                current_session.form.values
                if current_session is not None and current_session.form is not None
                else {}
            )
            other_series = "".join(
                char
                for char in str(
                    other_values.get(f"{other_role}.passport_series", "")
                )
                if char.isdigit()
            )
            other_number = "".join(
                char
                for char in str(
                    other_values.get(f"{other_role}.passport_number", "")
                )
                if char.isdigit()
            )
            if (
                selected_series
                and selected_number
                and selected_series == other_series
                and selected_number == other_number
            ):
                selected_name = "продавца" if passport_role == "seller" else "покупателя"
                other_name = "покупателя" if passport_role == "seller" else "продавца"
                await state.set_state(PhotoDealStates.choosing_document)
                await callback.message.edit_text(
                    f"⚠️ Этот паспорт уже сохранён как паспорт {other_name}. "
                    f"Похоже, при загрузке паспорта {selected_name} выбран не тот человек. "
                    "Выберите правильный документ и загрузите его заново.",
                    reply_markup=_document_keyboard(
                        deal_id,
                        revision,
                        current_session.confirmed_documents if current_session else set(),
                    ),
                )
                return
        logger.info(
            "OCR accepted fields kind=%s role=%s keys=%s uncertain=%s skipped_keys=%s",
            expected_kind.value,
            state_data.get("passport_role"),
            sorted(accepted),
            sorted(uncertain_fields),
            sorted(skipped_keys),
        )
        try:
            staged = await deal_store.stage_recognized_fields(
                callback.from_user.id,
                deal_id,
                accepted,
                revision,
                required_corrections,
                alternatives,
                uncertain_fields,
                expected_kind.value,
                state_data.get("passport_role")
                if isinstance(state_data.get("passport_role"), str)
                else None,
            )
        except (LookupError, ManualEntryError) as error:
            await state.set_state(PhotoDealStates.choosing_document)
            await callback.message.edit_text(
                str(error), reply_markup=_document_keyboard(deal_id, revision)
            )
            return
        await state.set_state(PhotoDealStates.reviewing)
        await state.set_data({"deal_id": deal_id, "revision": staged.revision})
        suffix = f"\n\nЕщё {skipped} полей нужно будет проверить вручную." if skipped else ""
        if staged.pending_ocr_required_corrections:
            labels = ", ".join(
                FIELD_BY_KEY[key].label
                for key in staged.pending_ocr_required_corrections
            )
            suffix += (
                "\n\n⚠️ Нужно обязательно уточнить: "
                f"{labels}. Нажмите «Изменить данные»."
            )
        document_title = _specific_document_title(
            staged.pending_ocr_document_kind,
            staged.pending_ocr_passport_role,
        )
        await callback.message.edit_text(
            f"Проверьте найденные данные · {document_title}:\n\n"
            + "\n".join(lines)
            + suffix,
            reply_markup=_recognized_keyboard(deal_id, staged.revision),
        )

    @router.message(PhotoDealStates.uploading)
    async def require_photo(message: Message) -> None:
        await message.answer("Отправьте фото как изображение или JPG/PNG-файл.")

    @router.callback_query(
        PhotoDealStates.reviewing, F.data.startswith("photo-edit:")
    )
    async def choose_recognized_field(
        callback: CallbackQuery, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "photo-edit", 2)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or not session.pending_ocr_fields
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await callback.answer()
        document_title = _specific_document_title(
            session.pending_ocr_document_kind,
            session.pending_ocr_passport_role,
        )
        await callback.message.edit_text(
            f"✏️ Изменение данных · {document_title}\n\n"
            "Выберите конкретное поле. Значком ⚠️ отмечены значения, "
            "которые распознаны неуверенно и требуют особой проверки.",
            reply_markup=_recognized_all_fields_keyboard(
                deal_id,
                session.revision,
                set(session.pending_ocr_fields),
                set(session.pending_ocr_uncertain_fields or ()),
            ),
        )

    @router.callback_query(
        PhotoDealStates.reviewing, F.data.startswith("photo-section:")
    )
    async def choose_recognized_section(
        callback: CallbackQuery, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "photo-section", 3)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, section = parts
        session = deal_store.get(callback.from_user.id)
        field_keys = set(session.pending_ocr_fields or {}) if session else set()
        section_label = dict(_SECTIONS).get(section)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or section_label is None
            or not any(_section_for(key) == section for key in field_keys)
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await callback.answer()
        await callback.message.edit_text(
            f"✏️ {section_label}\n\nВыберите поле, которое нужно изменить.",
            reply_markup=_recognized_fields_keyboard(
                section, deal_id, session.revision, field_keys
            ),
        )

    @router.callback_query(
        PhotoDealStates.reviewing, F.data.startswith("photo-field:")
    )
    async def begin_recognized_field_correction(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "photo-field", 3)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, raw_index = parts
        try:
            revision = int(raw_revision)
            spec = MANUAL_FIELDS[int(raw_index)]
        except (ValueError, IndexError):
            await callback.answer("Неизвестное поле", show_alert=True)
            return
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or session.revision != revision
            or not session.pending_ocr_fields
            or spec.key not in session.pending_ocr_fields
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        current_value = session.pending_ocr_fields[spec.key]
        previous_value = (
            session.form.values.get(spec.key)
            if session.form is not None and spec.key in session.form.values
            else None
        )
        value_text = f"Сейчас: {current_value}"
        field_alternatives = (
            session.pending_ocr_alternatives.get(spec.key, ())
            if session.pending_ocr_alternatives
            else ()
        )
        if field_alternatives:
            rendered_alternatives = "\n".join(
                f"{index}. {_text_preview(value, 280)}"
                for index, value in enumerate(field_alternatives, start=1)
            )
            value_text = f"На фото найдены разные варианты:\n{rendered_alternatives}"
        if (
            previous_value is not None
            and session.pending_ocr_required_corrections
            and spec.key in session.pending_ocr_required_corrections
        ):
            value_text = (
                f"Уже сохранено: {previous_value}\n"
                f"Найдено на новом фото: {current_value}"
            )
        await state.set_state(PhotoDealStates.correcting)
        await state.set_data(
            {
                "deal_id": deal_id,
                "revision": revision,
                "field_key": spec.key,
                "review_panel_chat_id": callback.message.chat.id,
                "review_panel_message_id": callback.message.message_id,
            }
        )
        await callback.answer()
        await callback.message.edit_text(
            f"✏️ Изменение поля\n\n"
            f"Документ: {_specific_document_title(session.pending_ocr_document_kind, session.pending_ocr_passport_role)}\n"
            f"Поле: {spec.label}\n"
            f"{value_text}\n\n"
            "Можно скопировать текущее значение кнопкой ниже, исправить "
            "нужную букву или цифру и отправить результат сообщением.\n\n"
            f"{spec.prompt}",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="📋 Скопировать текущее значение",
                            copy_text=CopyTextButton(text=current_value[:256]),
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            text="Назад к данным",
                            callback_data=f"photo-review:{deal_id}:{revision}",
                        )
                    ]
                ]
            ),
        )

    @router.message(PhotoDealStates.correcting)
    async def save_recognized_field_correction(
        message: Message, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        if message.from_user is None or message.text is None:
            await message.answer("Отправьте правильное значение обычным текстом.")
            return
        data = await state.get_data()
        deal_id = data.get("deal_id")
        revision = data.get("revision")
        field_key = data.get("field_key")
        if (
            not isinstance(deal_id, str)
            or not isinstance(revision, int)
            or not isinstance(field_key, str)
        ):
            await state.clear()
            await message.answer("Операция устарела. Начните заново через /start.")
            return
        try:
            session = await deal_store.update_pending_ocr_field(
                message.from_user.id,
                deal_id,
                field_key,
                message.text,
                revision,
            )
        except (LookupError, ManualEntryError) as error:
            await message.answer(str(error))
            return
        assert session.pending_ocr_fields is not None
        await state.set_state(PhotoDealStates.reviewing)
        panel_chat_id = data.get("review_panel_chat_id")
        panel_message_id = data.get("review_panel_message_id")
        await state.set_data({"deal_id": deal_id, "revision": session.revision})
        document_title = _specific_document_title(
            session.pending_ocr_document_kind,
            session.pending_ocr_passport_role,
        )
        updated_text = (
            f"✅ Данные обновлены · {document_title}.\n\n"
            + _pending_ocr_text(
                session.pending_ocr_fields,
                session.pending_ocr_required_corrections,
                session.pending_ocr_uncertain_fields,
            )
        )
        if isinstance(panel_chat_id, int) and isinstance(panel_message_id, int):
            try:
                await message.bot.edit_message_reply_markup(
                    chat_id=panel_chat_id,
                    message_id=panel_message_id,
                    reply_markup=None,
                )
            except TelegramBadRequest:
                pass
        await message.answer(
            updated_text,
            reply_markup=_recognized_keyboard(deal_id, session.revision),
        )

    @router.callback_query(F.data.startswith("photo-review:"))
    async def return_to_recognized_review(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "photo-review", 2)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or not session.pending_ocr_fields
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.set_state(PhotoDealStates.reviewing)
        await state.set_data({"deal_id": deal_id, "revision": session.revision})
        await callback.answer()
        document_title = _specific_document_title(
            session.pending_ocr_document_kind,
            session.pending_ocr_passport_role,
        )
        await callback.message.edit_text(
            f"Документ: {document_title}\n\n"
            + _pending_ocr_text(
                session.pending_ocr_fields,
                session.pending_ocr_required_corrections,
                session.pending_ocr_uncertain_fields,
            ),
            reply_markup=_recognized_keyboard(deal_id, session.revision),
        )

    @router.callback_query(PhotoDealStates.reviewing, F.data.startswith("photo-confirm:"))
    async def confirm_photo_fields(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "photo-confirm", 2)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision = parts
        try:
            session = await deal_store.confirm_recognized_fields(
                callback.from_user.id, deal_id, int(raw_revision)
            )
        except (ValueError, LookupError, ManualEntryError) as error:
            await callback.answer(str(error), show_alert=True)
            return
        await callback.answer()
        assert session.form is not None
        logger.info(
            "Confirmed document documents=%s missing_keys=%s",
            sorted(session.confirmed_documents),
            sorted(
                spec.key
                for spec in MANUAL_FIELDS
                if spec.key not in session.form.values
            ),
        )
        if session.form.next_missing() is None:
            try:
                deal = await deal_store.finalize_manual_form(
                    callback.from_user.id, deal_id
                )
            except (LookupError, ManualEntryError) as error:
                await callback.message.answer(str(error))
                return
            current = deal_store.get(callback.from_user.id)
            assert current is not None
            await state.set_state(ManualDealStates.review)
            await state.set_data(
                {"deal_id": deal_id, "revision": current.revision}
            )
            await callback.message.edit_text(
                "Все обязательные данные собраны.\n\n" + _review_text(deal),
                reply_markup=_review_keyboard(deal_id, current.revision),
            )
            return
        await state.set_state(PhotoDealStates.choosing_document)
        await state.set_data({"deal_id": deal_id, "revision": session.revision})
        await callback.message.edit_text(
            "Данные документа подтверждены.\n\n"
            + _deal_progress_text(session)
            + "\n\nМожно добавить следующий документ или заполнить недостающее вручную.",
            reply_markup=_document_keyboard(
                deal_id, session.revision, session.confirmed_documents
            ),
        )

    @router.callback_query(F.data.startswith("photo-retry:"))
    async def retry_photo(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "photo-retry", 2)
        session = deal_store.get(callback.from_user.id)
        if (
            parts is None
            or session is None
            or session.deal_id != parts[0]
            or str(session.revision) != parts[1]
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        try:
            session = await deal_store.discard_pending_recognized_fields(
                callback.from_user.id, session.deal_id, session.revision
            )
        except LookupError as error:
            await callback.answer(str(error), show_alert=True)
            return
        photo_batches.pop(callback.from_user.id, None)
        photo_batch_versions.pop(callback.from_user.id, None)
        await state.set_state(PhotoDealStates.choosing_document)
        await state.set_data({"deal_id": session.deal_id, "revision": session.revision})
        await callback.answer()
        if isinstance(callback.message, Message):
            await callback.message.edit_text(
                _deal_progress_text(session)
                + "\n\nВыберите документ и пришлите новое фото.",
                reply_markup=_document_keyboard(
                    session.deal_id,
                    session.revision,
                    session.confirmed_documents,
                ),
            )

    @router.callback_query(F.data.startswith("photo-manual:"))
    async def continue_photo_manually(
        callback: CallbackQuery, state: FSMContext, deal_store: InMemoryDealStore
    ) -> None:
        if not isinstance(callback.message, Message):
            return
        parts = _callback_parts(callback.data, "photo-manual", 2)
        session = deal_store.get(callback.from_user.id)
        if (
            parts is None
            or session is None
            or session.deal_id != parts[0]
            or str(session.revision) != parts[1]
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        try:
            session = await deal_store.discard_pending_recognized_fields(
                callback.from_user.id, session.deal_id, session.revision
            )
        except LookupError as error:
            await callback.answer(str(error), show_alert=True)
            return
        assert session.form is not None
        missing_keys = {
            spec.key for spec in MANUAL_FIELDS if spec.key not in session.form.values
        }
        if not missing_keys:
            try:
                deal = await deal_store.finalize_manual_form(
                    callback.from_user.id, session.deal_id
                )
            except (LookupError, ManualEntryError) as error:
                await callback.answer(str(error), show_alert=True)
                return
            current = deal_store.get(callback.from_user.id)
            assert current is not None
            await state.set_state(ManualDealStates.review)
            await state.set_data(
                {"deal_id": current.deal_id, "revision": current.revision}
            )
            await callback.answer()
            await callback.message.edit_text(
                _review_text(deal),
                reply_markup=_review_keyboard(current.deal_id, current.revision),
            )
            return
        data = await state.get_data()
        if isinstance(data.get("document_kind"), str):
            await state.set_state(PhotoDealStates.uploading)
        else:
            await state.set_state(PhotoDealStates.choosing_document)
        await state.update_data(
            deal_id=session.deal_id,
            revision=session.revision,
            selective_manual=False,
        )
        await callback.answer()
        await callback.message.edit_text(
            _deal_progress_text(session)
            + "\n\nВыберите конкретное недостающее поле. "
            "Уже подтверждённые данные сохранены.",
            reply_markup=_manual_missing_all_fields_keyboard(
                session.deal_id, session.revision, missing_keys
            ),
        )

    @router.callback_query(F.data.startswith("manual-section:"))
    async def choose_missing_manual_section(
        callback: CallbackQuery, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "manual-section", 3)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, section = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or session.form is None
            or section not in {key for key, _ in _SECTIONS}
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        missing_keys = {
            spec.key for spec in MANUAL_FIELDS if spec.key not in session.form.values
        }
        if not any(_section_for(key) == section for key in missing_keys):
            await callback.answer("В этом разделе всё уже заполнено", show_alert=True)
            return
        await callback.answer()
        await callback.message.edit_text(
            "Выберите поле, которое хотите заполнить.",
            reply_markup=_manual_missing_fields_keyboard(
                section, deal_id, session.revision, missing_keys
            ),
        )

    @router.callback_query(F.data.startswith("manual-field:"))
    async def choose_missing_manual_field(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "manual-field", 3)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision, raw_index = parts
        try:
            revision = int(raw_revision)
            spec = MANUAL_FIELDS[int(raw_index)]
        except (ValueError, IndexError):
            await callback.answer("Неизвестное поле", show_alert=True)
            return
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or session.revision != revision
            or session.form is None
            or spec.key in session.form.values
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.set_state(ManualDealStates.entering)
        await state.update_data(
            deal_id=deal_id,
            revision=revision,
            field_key=spec.key,
            edit_mode=False,
            selective_manual=True,
            manual_panel_chat_id=callback.message.chat.id,
            manual_panel_message_id=callback.message.message_id,
        )
        await callback.answer()
        await callback.message.edit_text(
            f"Заполнение поля «{spec.label}».\n\n{spec.prompt}"
        )

    @router.callback_query(F.data.startswith("manual-return:"))
    async def return_from_manual_picker(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "manual-return", 2)
        if parts is None or not isinstance(callback.message, Message):
            return
        deal_id, raw_revision = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        data = await state.get_data()
        await callback.answer()
        if isinstance(data.get("document_kind"), str):
            await state.set_state(PhotoDealStates.uploading)
            await render_upload_panel(
                callback.from_user.id, state, callback.bot, callback.message
            )
            return
        await state.set_state(PhotoDealStates.choosing_document)
        await callback.message.edit_text(
            _deal_progress_text(session) + "\n\nВыберите, какой документ добавим.",
            reply_markup=_document_keyboard(
                deal_id, session.revision, session.confirmed_documents
            ),
        )

    @router.message(Command("cancel"))
    async def cancel(
        message: Message,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        state_data = await state.get_data()
        if message.from_user is not None:
            photo_batches.pop(message.from_user.id, None)
            photo_batch_versions.pop(message.from_user.id, None)
        deal_id = state_data.get("deal_id")
        if message.from_user is not None and isinstance(deal_id, str):
            await deal_store.delete_if_current(message.from_user.id, deal_id)
        await state.clear()
        await message.answer("Активная операция отменена, временные данные удалены.")

    @router.message(ManualDealStates.entering)
    async def accept_manual_value(
        message: Message,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        if message.from_user is None or message.text is None:
            await message.answer("На этом шаге отправьте значение обычным текстом.")
            return
        state_data = await state.get_data()
        deal_id = state_data.get("deal_id")
        field_key = state_data.get("field_key")
        revision = state_data.get("revision")
        if (
            not isinstance(deal_id, str)
            or not isinstance(field_key, str)
            or not isinstance(revision, int)
        ):
            await state.clear()
            await message.answer("Операция потеряла состояние. Начните заново через /start.")
            return
        try:
            session = await deal_store.submit_manual_field(
                message.from_user.id,
                deal_id,
                field_key,
                message.text,
                revision,
            )
        except (LookupError, ManualEntryError) as error:
            await message.answer(str(error))
            return

        assert session.form is not None
        if state_data.get("selective_manual") is True:
            missing_keys = {
                spec.key
                for spec in MANUAL_FIELDS
                if spec.key not in session.form.values
            }
            if missing_keys:
                if isinstance(state_data.get("document_kind"), str):
                    await state.set_state(PhotoDealStates.uploading)
                else:
                    await state.set_state(PhotoDealStates.choosing_document)
                await state.update_data(
                    revision=session.revision,
                    field_key=None,
                    selective_manual=False,
                )
                text = (
                    "✅ Поле сохранено.\n\n"
                    + _deal_progress_text(session)
                    + "\n\nВыберите следующее поле или вернитесь к загрузке документов."
                )
                keyboard = _manual_missing_all_fields_keyboard(
                    deal_id, session.revision, missing_keys
                )
                panel_chat_id = state_data.get("manual_panel_chat_id")
                panel_message_id = state_data.get("manual_panel_message_id")
                if isinstance(panel_chat_id, int) and isinstance(panel_message_id, int):
                    try:
                        await message.bot.edit_message_reply_markup(
                            chat_id=panel_chat_id,
                            message_id=panel_message_id,
                            reply_markup=None,
                        )
                    except TelegramBadRequest:
                        pass
                panel = await message.answer(text, reply_markup=keyboard)
                await state.update_data(
                    manual_panel_chat_id=panel.chat.id,
                    manual_panel_message_id=panel.message_id,
                )
                return
            try:
                deal = await deal_store.finalize_manual_form(
                    message.from_user.id, deal_id
                )
            except (LookupError, ManualEntryError) as error:
                await message.answer(str(error))
                return
            current = deal_store.get(message.from_user.id)
            assert current is not None
            await state.set_state(ManualDealStates.review)
            await state.set_data(
                {"deal_id": deal_id, "revision": current.revision}
            )
            await message.answer(
                "Все обязательные поля заполнены.\n\n" + _review_text(deal),
                reply_markup=_review_keyboard(deal_id, current.revision),
            )
            return

        if state_data.get("edit_mode") is True:
            try:
                deal = await deal_store.finalize_manual_form(
                    message.from_user.id, deal_id
                )
            except (LookupError, ManualEntryError) as error:
                await message.answer(str(error))
                return
            await state.set_state(ManualDealStates.review)
            current = deal_store.get(message.from_user.id)
            assert current is not None
            await state.set_data(
                {"deal_id": deal_id, "revision": current.revision}
            )
            await message.answer(
                "Исправление сохранено.\n\n" + _review_text(deal),
                reply_markup=_review_keyboard(deal_id, current.revision),
            )
            return

        next_field = session.form.next_missing()
        if next_field is not None:
            await state.update_data(
                field_key=next_field.key, revision=session.revision
            )
            completed = len(session.form.values)
            await message.answer(
                f"Сохранено · следующий вопрос {completed + 1} из {len(MANUAL_FIELDS)}.\n\n"
                f"{next_field.prompt}"
            )
            return

        try:
            deal = await deal_store.finalize_manual_form(message.from_user.id, deal_id)
        except (LookupError, ManualEntryError) as error:
            await message.answer(str(error))
            return
        await state.set_state(ManualDealStates.review)
        current = deal_store.get(message.from_user.id)
        assert current is not None
        await state.set_data({"deal_id": deal_id, "revision": current.revision})
        await message.answer(
            _review_text(deal),
            reply_markup=_review_keyboard(deal_id, current.revision),
        )

    @router.callback_query(ManualDealStates.review, F.data.startswith("review-edit:"))
    async def choose_edit_section(
        callback: CallbackQuery, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "review-edit", 2)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or session.form is None
            or session.draft is None
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await callback.answer()
        if isinstance(callback.message, Message):
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass
            await callback.message.answer(
                "Выберите раздел, который хотите исправить.",
                reply_markup=_section_keyboard(deal_id, session.revision),
            )

    @router.callback_query(
        ManualDealStates.review, F.data.startswith("review-details:")
    )
    async def show_all_details(
        callback: CallbackQuery, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "review-details", 2)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or session.draft is None
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        if not isinstance(callback.message, Message):
            await callback.answer("Сообщение больше недоступно", show_alert=True)
            return
        await callback.answer()
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
        await callback.message.answer(
            "Ниже показаны полные персональные данные для проверки. "
            "Эти сообщения могут сохраняться в Telegram."
        )
        for section in _detailed_review_texts(session.draft):
            await callback.message.answer(section)
        await callback.message.answer(
            "Если всё верно, сформируйте PDF. Для исправления выберите нужный раздел.",
            reply_markup=_review_keyboard(deal_id, session.revision),
        )

    @router.callback_query(
        ManualDealStates.review, F.data.startswith("edit-section:")
    )
    async def choose_edit_field(
        callback: CallbackQuery, deal_store: InMemoryDealStore
    ) -> None:
        parts = _callback_parts(callback.data, "edit-section", 3)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision, section = parts
        session = deal_store.get(callback.from_user.id)
        if section not in {key for key, _ in _SECTIONS}:
            await callback.answer("Неизвестный раздел", show_alert=True)
            return
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await callback.answer()
        if isinstance(callback.message, Message):
            await callback.message.answer(
                "Выберите поле.",
                reply_markup=_field_keyboard(section, deal_id, session.revision),
            )

    @router.callback_query(ManualDealStates.review, F.data.startswith("edit-field:"))
    async def begin_field_edit(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "edit-field", 3)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision, raw_index = parts
        try:
            spec = MANUAL_FIELDS[int(raw_index)]
        except (ValueError, IndexError):
            await callback.answer("Неизвестное поле", show_alert=True)
            return
        data = await state.get_data()
        session = deal_store.get(callback.from_user.id)
        if (
            data.get("deal_id") != deal_id
            or str(data.get("revision")) != raw_revision
            or session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or session.form is None
            or session.draft is None
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.set_state(ManualDealStates.entering)
        await state.set_data(
            {
                "deal_id": deal_id,
                "revision": int(raw_revision),
                "field_key": spec.key,
                "edit_mode": True,
            }
        )
        await callback.answer()
        if isinstance(callback.message, Message):
            await callback.message.answer(f"Исправление поля «{spec.label}».\n\n{spec.prompt}")

    @router.callback_query(ManualDealStates.review, F.data.startswith("edit-back:"))
    async def return_to_review(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "edit-back", 2)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision = parts
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or str(session.revision) != raw_revision
            or session.draft is None
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await callback.answer()
        if isinstance(callback.message, Message):
            await callback.message.answer(
                _review_text(session.draft),
                reply_markup=_review_keyboard(deal_id, session.revision),
            )

    @router.callback_query(ManualDealStates.review, F.data.startswith("review-pdf:"))
    async def generate_pdf(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "review-pdf", 2)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision = parts
        try:
            revision = int(raw_revision)
        except ValueError:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        session = deal_store.get(callback.from_user.id)
        if (
            session is None
            or session.deal_id != deal_id
            or session.revision != revision
            or session.draft is None
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        if not isinstance(callback.message, Message):
            await callback.answer("Сообщение больше недоступно", show_alert=True)
            return
        if not await deal_store.begin_generation(
            callback.from_user.id, deal_id, revision
        ):
            await callback.answer("PDF уже формируется", show_alert=True)
            return
        await callback.answer("Формирую PDF…")
        renderer = PdfRenderer.from_defaults(settings)
        try:
            with tempfile.TemporaryDirectory(prefix=f"deal-{session.deal_id}-") as directory:
                output = Path(directory) / "dogovor-kupli-prodazhi.pdf"
                await asyncio.to_thread(renderer.render, session.draft, output)
                async with deal_store.lock_for(callback.from_user.id):
                    current = deal_store.get(callback.from_user.id)
                    if (
                        current is None
                        or current.deal_id != deal_id
                        or current.revision != revision
                    ):
                        return
                    try:
                        await callback.message.edit_reply_markup(reply_markup=None)
                    except TelegramBadRequest:
                        pass
                    await callback.message.answer_document(
                        FSInputFile(output),
                        caption=(
                            "Договор сформирован. Проверьте его перед печатью. "
                            "После проверки можно исправить данные или завершить сделку."
                        ),
                        reply_markup=_after_pdf_keyboard(deal_id, revision),
                    )
        except FieldOverflowError as error:
            label = _PDF_FIELD_LABELS.get(error.field_name, error.field_name)
            await callback.message.answer(
                f"Значение поля «{label}» слишком длинное для строки PDF. "
                "Сократите его без потери смысла и попробуйте ещё раз.",
                reply_markup=_review_keyboard(deal_id, revision),
            )
        except PdfRenderError:
            await callback.message.answer(
                "Не удалось сформировать PDF. Проверьте заполненные данные "
                "и попробуйте ещё раз.",
                reply_markup=_review_keyboard(deal_id, revision),
            )
        finally:
            await deal_store.finish_generation(
                callback.from_user.id, deal_id, revision
            )

    @router.callback_query(ManualDealStates.review, F.data.startswith("review-new:"))
    async def start_new_deal_after_pdf(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "review-new", 2)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision = parts
        try:
            revision = int(raw_revision)
        except ValueError:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        if not await deal_store.delete_if_current(
            callback.from_user.id, deal_id, revision
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.clear()
        await callback.answer("Предыдущая сделка удалена")
        if isinstance(callback.message, Message):
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass
            await callback.message.answer(
                "Начинаем новую сделку. Выберите способ заполнения.",
                reply_markup=_start_keyboard(),
            )

    @router.callback_query(ManualDealStates.review, F.data.startswith("review-finish:"))
    async def finish_deal(
        callback: CallbackQuery,
        state: FSMContext,
        deal_store: InMemoryDealStore,
    ) -> None:
        parts = _callback_parts(callback.data, "review-finish", 2)
        if parts is None:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        deal_id, raw_revision = parts
        try:
            revision = int(raw_revision)
        except ValueError:
            await callback.answer("Некорректная кнопка", show_alert=True)
            return
        if not await deal_store.delete_if_current(
            callback.from_user.id, deal_id, revision
        ):
            await callback.answer("Эта кнопка устарела", show_alert=True)
            return
        await state.clear()
        await callback.answer()
        if isinstance(callback.message, Message):
            await callback.message.answer(
                "Сделка завершена, временные данные удалены.",
                reply_markup=_start_keyboard(),
            )

    dispatcher.include_router(router)
    return dispatcher


async def run_bot(settings: Settings) -> None:
    if settings.telegram_bot_token is None or not settings.telegram_bot_token.get_secret_value():
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    if not settings.allowed_telegram_ids:
        raise RuntimeError("ALLOWED_TELEGRAM_IDS must not be empty")
    bot = Bot(settings.telegram_bot_token.get_secret_value())
    deal_store = InMemoryDealStore(timedelta(minutes=settings.deal_ttl_minutes))
    dispatcher = build_dispatcher(settings, deal_store)

    async def purge_expired_sessions() -> None:
        interval = min(60.0, max(5.0, settings.deal_ttl_minutes * 30.0))
        while True:
            await asyncio.sleep(interval)
            await deal_store.purge_expired_async()

    janitor = asyncio.create_task(purge_expired_sessions())
    try:
        await dispatcher.start_polling(bot)
    finally:
        janitor.cancel()
        with suppress(asyncio.CancelledError):
            await janitor
        deal_store.clear()
        await bot.session.close()

