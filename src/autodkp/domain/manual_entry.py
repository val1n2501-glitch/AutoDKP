from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Callable

from pydantic import ValidationError

from .models import DealDraft, Person, Vehicle, VehicleDocument


class ManualEntryError(ValueError):
    """A user-facing validation error that never contains the submitted value."""


Parser = Callable[[str], object]


def _required_text(raw: str) -> str:
    value = " ".join(raw.strip().split())
    if not value:
        raise ManualEntryError("Значение не может быть пустым.")
    return value


def _date(raw: str) -> date:
    try:
        return datetime.strptime(raw.strip(), "%d.%m.%Y").date()
    except ValueError as error:
        raise ManualEntryError("Введите дату в формате ДД.ММ.ГГГГ.") from error


def _integer(raw: str) -> int:
    compact = raw.replace(" ", "").strip()
    if not compact.isdecimal():
        raise ManualEntryError("Введите целое число без дополнительных символов.")
    return int(compact)


def _positive_integer(raw: str) -> int:
    value = _integer(raw)
    if value <= 0:
        raise ManualEntryError("Введите число больше нуля.")
    return value


def _nonnegative_integer(raw: str) -> int:
    value = _integer(raw)
    if value < 0:
        raise ManualEntryError("Значение не может быть отрицательным.")
    return value


def _manufacture_year(raw: str) -> int:
    value = _integer(raw)
    if value < 1900 or value > date.today().year + 1:
        raise ManualEntryError(
            f"Введите год от 1900 до {date.today().year + 1}."
        )
    return value


def _decimal(raw: str) -> Decimal:
    compact = raw.replace(" ", "").replace(",", ".").strip()
    try:
        value = Decimal(compact)
    except InvalidOperation as error:
        raise ManualEntryError("Введите число, например 181 или 181,5.") from error
    if not value.is_finite():
        raise ManualEntryError("Введите конечное число.")
    return value


def _positive_decimal(raw: str) -> Decimal:
    value = _decimal(raw)
    if value <= 0:
        raise ManualEntryError("Введите число больше нуля.")
    return value


def _digits(length: int, label: str) -> Parser:
    def parse(raw: str) -> str:
        value = raw.replace(" ", "").strip()
        if len(value) != length or not value.isdecimal():
            suffix = "цифры" if length == 4 else "цифр"
            raise ManualEntryError(f"{label}: требуется ровно {length} {suffix}.")
        return value

    return parse


def _vin(raw: str) -> str:
    value = raw.replace(" ", "").upper()
    allowed = set("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
    if len(value) != 17 or any(char not in allowed for char in value):
        raise ManualEntryError(
            "VIN должен содержать 17 латинских букв и цифр без I, O и Q."
        )
    return value


@dataclass(frozen=True, slots=True)
class ManualFieldSpec:
    key: str
    label: str
    prompt: str
    parser: Parser = _required_text


MANUAL_FIELDS: tuple[ManualFieldSpec, ...] = (
    ManualFieldSpec("agreement_place", "Место составления", "Введите место составления договора."),
    ManualFieldSpec(
        "agreement_date",
        "Дата договора",
        "Введите дату договора в формате ДД.ММ.ГГГГ.",
        _date,
    ),
    ManualFieldSpec("buyer.full_name", "ФИО покупателя", "Введите ФИО покупателя полностью."),
    ManualFieldSpec(
        "buyer.birth_date", "Дата рождения покупателя", "Введите дату рождения покупателя.", _date
    ),
    ManualFieldSpec("buyer.birth_place", "Место рождения покупателя", "Введите место рождения покупателя."),
    ManualFieldSpec("buyer.address", "Адрес покупателя", "Введите адрес места жительства покупателя."),
    ManualFieldSpec(
        "buyer.passport_series",
        "Серия паспорта покупателя",
        "Введите 4 цифры серии паспорта покупателя.",
        _digits(4, "Серия паспорта"),
    ),
    ManualFieldSpec(
        "buyer.passport_number",
        "Номер паспорта покупателя",
        "Введите 6 цифр номера паспорта покупателя.",
        _digits(6, "Номер паспорта"),
    ),
    ManualFieldSpec("buyer.passport_issued_by", "Кем выдан паспорт покупателя", "Введите, кем выдан паспорт покупателя."),
    ManualFieldSpec(
        "buyer.passport_issue_date",
        "Дата выдачи паспорта покупателя",
        "Введите дату выдачи паспорта покупателя.",
        _date,
    ),
    ManualFieldSpec("seller.full_name", "ФИО продавца", "Введите ФИО продавца полностью."),
    ManualFieldSpec(
        "seller.birth_date", "Дата рождения продавца", "Введите дату рождения продавца.", _date
    ),
    ManualFieldSpec("seller.birth_place", "Место рождения продавца", "Введите место рождения продавца."),
    ManualFieldSpec("seller.address", "Адрес продавца", "Введите адрес места жительства продавца."),
    ManualFieldSpec(
        "seller.passport_series",
        "Серия паспорта продавца",
        "Введите 4 цифры серии паспорта продавца.",
        _digits(4, "Серия паспорта"),
    ),
    ManualFieldSpec(
        "seller.passport_number",
        "Номер паспорта продавца",
        "Введите 6 цифр номера паспорта продавца.",
        _digits(6, "Номер паспорта"),
    ),
    ManualFieldSpec("seller.passport_issued_by", "Кем выдан паспорт продавца", "Введите, кем выдан паспорт продавца."),
    ManualFieldSpec(
        "seller.passport_issue_date",
        "Дата выдачи паспорта продавца",
        "Введите дату выдачи паспорта продавца.",
        _date,
    ),
    ManualFieldSpec("vehicle.make_model", "Марка и модель", "Введите марку и модель автомобиля."),
    ManualFieldSpec("vehicle.vin", "VIN", "Введите 17-значный VIN автомобиля.", _vin),
    ManualFieldSpec("vehicle.vehicle_type", "Тип ТС", "Введите тип транспортного средства."),
    ManualFieldSpec(
        "vehicle.manufacture_year", "Год выпуска", "Введите год выпуска.", _manufacture_year
    ),
    ManualFieldSpec(
        "vehicle.mileage_km", "Пробег, км", "Введите пробег в километрах.", _nonnegative_integer
    ),
    ManualFieldSpec(
        "vehicle.engine_power_hp",
        "Мощность двигателя",
        "Введите мощность двигателя в л. с.",
        _positive_decimal,
    ),
    ManualFieldSpec(
        "vehicle.engine_displacement_cc",
        "Рабочий объём двигателя",
        "Введите рабочий объём двигателя в куб. см.",
        _positive_integer,
    ),
    ManualFieldSpec("vehicle.body_color", "Цвет кузова", "Введите цвет кузова."),
    ManualFieldSpec("vehicle.engine_model", "Модель двигателя", "Введите модель двигателя."),
    ManualFieldSpec("vehicle.engine_number", "Номер двигателя", "Введите номер двигателя."),
    ManualFieldSpec("vehicle.chassis_number", "Номер шасси (рамы)", "Введите номер шасси или слово «отсутствует»."),
    ManualFieldSpec("vehicle.body_number", "Номер кузова", "Введите номер кузова."),
    ManualFieldSpec("vehicle.registration_plate", "Государственный номер", "Введите государственный регистрационный номер."),
    ManualFieldSpec("vehicle.pts.series_number", "Серия и номер ПТС", "Введите серию и номер ПТС."),
    ManualFieldSpec("vehicle.pts.issued_by", "Кем выдан ПТС", "Введите, кем выдан ПТС."),
    ManualFieldSpec("vehicle.pts.issue_date", "Дата выдачи ПТС", "Введите дату выдачи ПТС.", _date),
    ManualFieldSpec("vehicle.sts.series_number", "Серия и номер СТС", "Введите серию и номер СТС."),
    ManualFieldSpec("vehicle.sts.issued_by", "Кем выдано СТС", "Введите, кем выдано СТС."),
    ManualFieldSpec("vehicle.sts.issue_date", "Дата выдачи СТС", "Введите дату выдачи СТС.", _date),
    ManualFieldSpec(
        "price_rubles",
        "Стоимость автомобиля",
        "Введите стоимость автомобиля в рублях.",
        _positive_integer,
    ),
)


FIELD_BY_KEY = {item.key: item for item in MANUAL_FIELDS}


@dataclass(slots=True)
class ManualDealForm:
    values: dict[str, object] = field(default_factory=dict)

    def next_missing(self) -> ManualFieldSpec | None:
        return next((item for item in MANUAL_FIELDS if item.key not in self.values), None)

    def submit(self, key: str, raw_value: str) -> object:
        try:
            spec = FIELD_BY_KEY[key]
        except KeyError as error:
            raise ManualEntryError("Неизвестное поле договора.") from error
        value = spec.parser(raw_value)
        self.values[key] = value
        return value

    def submit_many(self, raw_values: dict[str, str]) -> dict[str, object]:
        """Validate a confirmed OCR batch atomically before changing the form."""
        parsed: dict[str, object] = {}
        for key, raw_value in raw_values.items():
            try:
                spec = FIELD_BY_KEY[key]
            except KeyError as error:
                raise ManualEntryError("Неизвестное поле договора.") from error
            parsed[key] = spec.parser(raw_value)
        self.values.update(parsed)
        return parsed

    def build_deal(self) -> DealDraft:
        missing = [item.label for item in MANUAL_FIELDS if item.key not in self.values]
        if missing:
            raise ManualEntryError(f"Остались незаполненные поля: {len(missing)}.")

        def person(prefix: str) -> Person:
            return Person(
                full_name=self.values[f"{prefix}.full_name"],
                birth_date=self.values[f"{prefix}.birth_date"],
                birth_place=self.values[f"{prefix}.birth_place"],
                address=self.values[f"{prefix}.address"],
                passport_series=self.values[f"{prefix}.passport_series"],
                passport_number=self.values[f"{prefix}.passport_number"],
                passport_issued_by=self.values[f"{prefix}.passport_issued_by"],
                passport_issue_date=self.values[f"{prefix}.passport_issue_date"],
            )

        try:
            return DealDraft(
                agreement_place=self.values["agreement_place"],
                agreement_date=self.values["agreement_date"],
                buyer=person("buyer"),
                seller=person("seller"),
                vehicle=Vehicle(
                    make_model=self.values["vehicle.make_model"],
                    vin=self.values["vehicle.vin"],
                    vehicle_type=self.values["vehicle.vehicle_type"],
                    manufacture_year=self.values["vehicle.manufacture_year"],
                    mileage_km=self.values["vehicle.mileage_km"],
                    engine_power_hp=self.values["vehicle.engine_power_hp"],
                    engine_displacement_cc=self.values["vehicle.engine_displacement_cc"],
                    body_color=self.values["vehicle.body_color"],
                    engine_model=self.values["vehicle.engine_model"],
                    engine_number=self.values["vehicle.engine_number"],
                    chassis_number=self.values["vehicle.chassis_number"],
                    body_number=self.values["vehicle.body_number"],
                    registration_plate=self.values["vehicle.registration_plate"],
                    pts=VehicleDocument(
                        series_number=self.values["vehicle.pts.series_number"],
                        issued_by=self.values["vehicle.pts.issued_by"],
                        issue_date=self.values["vehicle.pts.issue_date"],
                    ),
                    sts=VehicleDocument(
                        series_number=self.values["vehicle.sts.series_number"],
                        issued_by=self.values["vehicle.sts.issued_by"],
                        issue_date=self.values["vehicle.sts.issue_date"],
                    ),
                ),
                price_rubles=self.values["price_rubles"],
            )
        except ValidationError as error:
            first = error.errors(include_input=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            raise ManualEntryError(f"Проверьте поле «{location}»: {first['msg']}.") from error

    def summary(self) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        for spec in MANUAL_FIELDS:
            if spec.key not in self.values:
                continue
            value = self.values[spec.key]
            if isinstance(value, date):
                rendered = value.strftime("%d.%m.%Y")
            elif isinstance(value, Decimal):
                rendered = format(value, "f")
            else:
                rendered = str(value)
            rows.append((spec.label, rendered))
        return tuple(rows)
