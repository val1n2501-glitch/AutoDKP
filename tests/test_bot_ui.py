from autodkp.bot.app import (
    _pending_ocr_text,
    _recognized_fields_keyboard,
    _recognized_all_fields_keyboard,
    _recognized_keyboard,
    _recognized_sections_keyboard,
    _detailed_review_texts,
    _deal_progress_text,
    _document_keyboard,
    _field_keyboard,
    _manual_missing_fields_keyboard,
    _manual_missing_all_fields_keyboard,
    _manual_missing_sections_keyboard,
    _review_keyboard,
    _review_text,
    _section_keyboard,
    _start_keyboard,
    _upload_keyboard,
    _upload_summary_text,
    _after_pdf_keyboard,
    build_dispatcher,
)
from autodkp.config import Settings
from autodkp.domain.demo import make_demo_deal
from autodkp.domain.manual_entry import MANUAL_FIELDS
from autodkp.domain.models import DocumentKind
from autodkp.storage.session_store import InMemoryDealStore
from datetime import timedelta


def test_dispatcher_can_be_built_without_telegram_token() -> None:
    dispatcher = build_dispatcher(Settings(_env_file=None))
    assert dispatcher.resolve_used_update_types() == ["callback_query", "message"]


def test_start_offers_photo_and_manual_flows() -> None:
    callbacks = [
        button.callback_data
        for row in _start_keyboard().inline_keyboard
        for button in row
    ]
    assert callbacks == ["deal:photo", "deal:new"]


def test_document_picker_marks_confirmed_documents_and_progress_explains_pdf() -> None:
    keyboard = _document_keyboard(
        "abc123",
        7,
        {"paper_pts", "passport:seller", "passport:buyer"},
    )
    texts = [button.text for row in keyboard.inline_keyboard for button in row]
    assert "✅ 🔄 Заменить ПТС" in texts
    assert "✅ 🔄 Заменить Паспорт продавца" in texts
    assert "✅ 🔄 Заменить Паспорт покупателя" in texts
    assert "СТС автомобиля" in texts

    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    progress = _deal_progress_text(session)
    assert f"Заполнено полей: 0 из {len(MANUAL_FIELDS)}" in progress
    assert "PDF пока недоступен" in progress


def test_review_masks_passports() -> None:
    deal = make_demo_deal()
    text = _review_text(deal)
    assert deal.buyer.full_name in text
    assert deal.seller.full_name in text
    assert f"{deal.buyer.passport_series} {deal.buyer.passport_number}" not in text
    assert f"{deal.seller.passport_series} {deal.seller.passport_number}" not in text
    assert "••••78" in text
    assert "••••56" in text


def test_pdf_result_offers_new_deal() -> None:
    callbacks = [
        button.callback_data
        for row in _after_pdf_keyboard("abc123", 7).inline_keyboard
        for button in row
    ]
    assert "review-new:abc123:7" in callbacks


def test_edit_keyboards_use_short_callback_payloads() -> None:
    keyboards = [
        _review_keyboard("abc123", 7),
        _section_keyboard("abc123", 7),
        _field_keyboard("vehicle", "abc123", 7),
    ]
    callbacks = [
        button.callback_data
        for keyboard in keyboards
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data is not None
    ]
    assert callbacks
    assert all(len(value.encode("utf-8")) <= 64 for value in callbacks)
    assert all("abc123:7" in value or value == "deal:new" for value in callbacks)


def test_photo_upload_uses_one_numbered_control_panel() -> None:
    text = _upload_summary_text(DocumentKind.STS.value, 3)
    assert "Загружено фото: 3 из 4" in text
    assert all(f"Фото {index} — готово" in text for index in (1, 2, 3))

    keyboard = _upload_keyboard("abc123", 7, 3)
    button_texts = [button.text for row in keyboard.inline_keyboard for button in row]
    callbacks = [
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data is not None
    ]
    assert all(
        any(f"Заменить фото {index}" in text for text in button_texts)
        for index in (1, 2, 3)
    )
    assert all(
        any(f"Удалить {index}" in text for text in button_texts)
        for index in (1, 2, 3)
    )
    assert "🔎 Распознать 3 фото" in button_texts
    assert all(len(value.encode("utf-8")) <= 64 for value in callbacks)


def test_empty_photo_panel_has_no_recognition_button() -> None:
    callbacks = [
        button.callback_data
        for row in _upload_keyboard("abc123", 7, 0).inline_keyboard
        for button in row
        if button.callback_data is not None
    ]
    assert not any(value.startswith("pg:") for value in callbacks)


def test_manual_supplement_starts_with_section_and_exact_field_choice() -> None:
    missing_keys = {"agreement_place", "vehicle.vin", "vehicle.sts.number"}
    sections = _manual_missing_sections_keyboard("abc123", 7, missing_keys)
    section_texts = [button.text for row in sections.inline_keyboard for button in row]
    assert "Договор" in section_texts
    assert "Автомобиль" in section_texts
    assert "СТС" in section_texts
    assert "Продавец" not in section_texts

    fields = _manual_missing_fields_keyboard(
        "vehicle", "abc123", 7, missing_keys
    )
    field_buttons = [button for row in fields.inline_keyboard for button in row]
    vin = next(spec for spec in MANUAL_FIELDS if spec.key == "vehicle.vin")
    assert vin.label in {button.text for button in field_buttons}
    assert all(
        len(button.callback_data.encode("utf-8")) <= 64
        for button in field_buttons
        if button.callback_data is not None
    )


def test_manual_supplement_flat_list_matches_exact_missing_fields() -> None:
    missing_keys = {
        "agreement_place",
        "agreement_date",
        "vehicle.vin",
        "vehicle.mileage_km",
        "price_rubles",
    }
    keyboard = _manual_missing_all_fields_keyboard("abc123", 7, missing_keys)
    labels = {
        button.text
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("manual-field:")
    }
    assert labels == {
        spec.label for spec in MANUAL_FIELDS if spec.key in missing_keys
    }


def test_recognized_data_editor_uses_russian_sections_and_field_names() -> None:
    field_keys = {spec.key for spec in MANUAL_FIELDS}
    sections = _recognized_sections_keyboard("abc123", 7, field_keys)
    section_texts = [button.text for row in sections.inline_keyboard for button in row]
    assert {"Договор", "Продавец", "Покупатель", "Автомобиль", "ПТС", "СТС", "Цена"} <= set(section_texts)

    shown_fields = []
    for section in ("deal", "seller", "buyer", "vehicle", "pts", "sts", "price"):
        keyboard = _recognized_fields_keyboard(section, "abc123", 7, field_keys)
        shown_fields.extend(
            button.text
            for row in keyboard.inline_keyboard
            for button in row
            if button.callback_data and button.callback_data.startswith("photo-field:")
        )
    assert sorted(shown_fields) == sorted(spec.label for spec in MANUAL_FIELDS)


def test_photo_review_calls_action_change_data() -> None:
    texts = [
        button.text
        for row in _recognized_keyboard("abc123", 7).inline_keyboard
        for button in row
    ]
    assert "✏️ Изменить данные" in texts
    assert "Проверьте исправленные данные" in _pending_ocr_text(
        {"vehicle.vin": "XW7BF4FK90S123456"}
    )
    uncertain_text = _pending_ocr_text(
        {
            "vehicle.vin": "XW7BF4FK90S123456",
            "vehicle.engine_number": "30492167",
        },
        uncertain_fields={"vehicle.engine_number"},
    )
    assert "⚠️ Номер двигателя: 30492167" in uncertain_text
    assert "распознаны неуверенно" in uncertain_text


def test_recognized_editor_lists_every_field_and_marks_uncertain_values() -> None:
    field_keys = {
        "vehicle.make_model",
        "vehicle.vin",
        "vehicle.sts.series_number",
    }
    keyboard = _recognized_all_fields_keyboard(
        "abc123", 7, field_keys, {"vehicle.vin"}
    )
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    labels = {button.text for button in buttons}
    assert "Марка и модель" in labels
    assert "⚠️ VIN" in labels
    assert "Серия и номер СТС" in labels
    field_callbacks = [
        button.callback_data
        for button in buttons
        if button.callback_data and button.callback_data.startswith("photo-field:")
    ]
    assert len(field_callbacks) == len(field_keys)
    assert all(len(value.encode("utf-8")) <= 64 for value in field_callbacks)


def test_detailed_review_contains_every_contract_value() -> None:
    deal = make_demo_deal()
    text = "\n".join(_detailed_review_texts(deal))
    expected = (
        deal.agreement_place,
        deal.buyer.full_name,
        deal.buyer.birth_place,
        deal.buyer.address,
        deal.buyer.passport_series,
        deal.buyer.passport_number,
        deal.buyer.passport_issued_by,
        deal.seller.full_name,
        deal.seller.birth_place,
        deal.seller.address,
        deal.seller.passport_series,
        deal.seller.passport_number,
        deal.seller.passport_issued_by,
        deal.vehicle.make_model,
        deal.vehicle.vin,
        deal.vehicle.vehicle_type,
        deal.vehicle.body_color,
        deal.vehicle.engine_model,
        deal.vehicle.engine_number,
        deal.vehicle.chassis_number,
        deal.vehicle.body_number,
        deal.vehicle.registration_plate,
        deal.vehicle.pts.series_number,
        deal.vehicle.pts.issued_by,
        deal.vehicle.sts.series_number,
        deal.vehicle.sts.issued_by,
    )
    assert all(value is not None and str(value) in text for value in expected)
