from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FieldStatus(StrEnum):
    MISSING = "missing"
    CANDIDATE = "candidate"
    CONFLICT = "conflict"
    CONFIRMED = "confirmed"
    NOT_APPLICABLE = "not_applicable"


class DocumentKind(StrEnum):
    PASSPORT = "passport"
    STS = "sts"
    PAPER_PTS = "paper_pts"
    EPTS_EXTRACT = "epts_extract"
    TEXT = "text"
    UNKNOWN = "unknown"


class SourceRef(BaseModel):
    document_id: str
    document_kind: DocumentKind
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    provider: str = "manual"
    provider_version: str | None = None


T = TypeVar("T")


class FieldCandidate(BaseModel, Generic[T]):
    raw_value: str
    normalized_value: T | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: SourceRef
    warnings: list[str] = Field(default_factory=list)


class ResolvedField(BaseModel, Generic[T]):
    value: T | None = None
    status: FieldStatus = FieldStatus.MISSING
    candidates: list[FieldCandidate[T]] = Field(default_factory=list)
    revision: int = Field(default=0, ge=0)

    @field_validator("value")
    @classmethod
    def strip_strings(cls, value: T | None) -> T | None:
        if isinstance(value, str):
            return value.strip()  # type: ignore[return-value]
        return value


class Person(BaseModel):
    full_name: str
    birth_date: date
    birth_place: str
    address: str
    passport_series: str
    passport_number: str
    passport_issued_by: str
    passport_issue_date: date


class VehicleDocument(BaseModel):
    series_number: str
    issued_by: str
    issue_date: date


class Vehicle(BaseModel):
    make_model: str
    vin: str | None = None
    vehicle_type: str
    manufacture_year: int = Field(ge=1900, le=2100)
    mileage_km: int | None = Field(default=None, ge=0)
    engine_power_hp: Decimal = Field(ge=0)
    engine_displacement_cc: int = Field(ge=0)
    body_color: str
    engine_model: str | None = None
    engine_number: str | None = None
    chassis_number: str | None = None
    body_number: str | None = None
    registration_plate: str | None = None
    pts: VehicleDocument
    sts: VehicleDocument

    @field_validator("vin")
    @classmethod
    def normalize_vin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace(" ", "").upper()
        if not normalized:
            return None
        return normalized

    @field_validator("body_number")
    @classmethod
    def normalize_body_number(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = "".join(value.split()).upper()
        return normalized or None


class DealDraft(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    agreement_place: str
    agreement_date: date
    buyer: Person
    seller: Person
    vehicle: Vehicle
    price_rubles: int = Field(gt=0, le=999_999_999_999)

