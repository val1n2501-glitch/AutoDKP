from __future__ import annotations

from typing import Protocol

from autodkp.domain.models import DocumentKind

from .models import RecognizedDocument


class RecognitionProvider(Protocol):
    name: str

    async def recognize(
        self,
        document_kind: DocumentKind,
        images: list[bytes],
        *,
        passport_role: str | None = None,
    ) -> RecognizedDocument: ...
