from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autodkp.domain.manual_entry import FIELD_BY_KEY
from autodkp.domain.models import DocumentKind


class RecognizedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str = Field(max_length=300)
    confidence: float = Field(ge=0, le=1)
    warning: str | None = Field(default=None, max_length=300)

    @field_validator("key")
    @classmethod
    def known_contract_field(cls, value: str) -> str:
        if value not in FIELD_BY_KEY:
            raise ValueError("unknown contract field")
        return value

    @field_validator("value")
    @classmethod
    def nonempty_value(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("empty value")
        return normalized


class RecognizedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_kind: DocumentKind
    fields: list[RecognizedField] = Field(default_factory=list)
    quality_warning: str | None = Field(default=None, max_length=500)
