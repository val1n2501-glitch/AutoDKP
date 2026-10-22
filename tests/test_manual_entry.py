from datetime import date

import pytest

from autodkp.domain.demo import make_demo_deal
from autodkp.domain.manual_entry import (
    MANUAL_FIELDS,
    ManualDealForm,
    ManualEntryError,
)


def _answers_from_demo() -> dict[str, str]:
    deal = make_demo_deal()
    answers: dict[str, str] = {
        "agreement_place": deal.agreement_place,
        "agreement_date": deal.agreement_date.strftime("%d.%m.%Y"),
        "price_rubles": str(deal.price_rubles),
    }
    for role in ("buyer", "seller"):
        person = getattr(deal, role)
        for name in (
            "full_name",
            "birth_place",
            "address",
            "passport_series",
            "passport_number",
            "passport_issued_by",
        ):
            answers[f"{role}.{name}"] = str(getattr(person, name))
        answers[f"{role}.birth_date"] = person.birth_date.strftime("%d.%m.%Y")
        answers[f"{role}.passport_issue_date"] = person.passport_issue_date.strftime("%d.%m.%Y")

    vehicle = deal.vehicle
    for name in (
        "make_model",
        "vin",
        "vehicle_type",
        "manufacture_year",
        "mileage_km",
        "engine_power_hp",
        "engine_displacement_cc",
        "body_color",
        "engine_model",
        "engine_number",
        "chassis_number",
        "body_number",
        "registration_plate",
    ):
        answers[f"vehicle.{name}"] = str(getattr(vehicle, name))
    for document_name in ("pts", "sts"):
        document = getattr(vehicle, document_name)
        answers[f"vehicle.{document_name}.series_number"] = document.series_number
        answers[f"vehicle.{document_name}.issued_by"] = document.issued_by
        answers[f"vehicle.{document_name}.issue_date"] = document.issue_date.strftime("%d.%m.%Y")
    return answers


def test_full_manual_entry_builds_the_same_deal_as_demo() -> None:
    form = ManualDealForm()
    answers = _answers_from_demo()
    for field in MANUAL_FIELDS:
        form.submit(field.key, answers[field.key])

    assert form.next_missing() is None
    assert form.build_deal() == make_demo_deal()


def test_next_missing_and_correction_do_not_lose_other_fields() -> None:
    form = ManualDealForm()
    form.submit("agreement_place", " г.   Москва ")
    form.submit("agreement_date", "21.10.2026")
    form.submit("agreement_place", "г. Тверь")

    assert form.values["agreement_place"] == "г. Тверь"
    assert form.values["agreement_date"] == date(2026, 10, 21)
    assert form.next_missing() == MANUAL_FIELDS[2]


def test_invalid_date_is_rejected_without_storing_raw_value() -> None:
    form = ManualDealForm()
    with pytest.raises(ManualEntryError, match="ДД.ММ.ГГГГ"):
        form.submit("agreement_date", "2026-10-21")
    assert "agreement_date" not in form.values


def test_incomplete_form_cannot_build_contract() -> None:
    with pytest.raises(ManualEntryError, match="незаполненные поля"):
        ManualDealForm().build_deal()


def test_price_accepts_spaces_and_summary_contains_no_internal_metadata() -> None:
    form = ManualDealForm()
    form.submit("price_rubles", "2 350 000")
    assert form.summary() == (("Стоимость автомобиля", "2350000"),)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("buyer.passport_series", "45A2", "4 цифры"),
        ("seller.passport_number", "12345", "6 цифр"),
        ("vehicle.vin", "КОРОТКИЙ", "17 латинских"),
        ("vehicle.mileage_km", "-1", "целое число"),
        ("vehicle.engine_power_hp", "0", "больше нуля"),
        ("price_rubles", "0", "больше нуля"),
    ],
)
def test_invalid_manual_values_are_rejected_immediately(
    key: str, value: str, message: str
) -> None:
    form = ManualDealForm()
    with pytest.raises(ManualEntryError, match=message):
        form.submit(key, value)
    assert key not in form.values
