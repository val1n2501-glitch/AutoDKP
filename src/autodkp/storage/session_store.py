from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Callable
from uuid import uuid4

from autodkp.domain.models import DealDraft
from autodkp.domain.manual_entry import FIELD_BY_KEY, ManualDealForm, ManualEntryError


@dataclass
class DealSession:
    deal_id: str
    owner_id: int
    created_at: datetime
    expires_at: datetime
    revision: int = 0
    draft: DealDraft | None = None
    form: ManualDealForm | None = None
    pending_ocr_fields: dict[str, str] | None = None
    pending_ocr_required_corrections: set[str] | None = None
    pending_ocr_uncertain_fields: set[str] | None = None
    pending_ocr_alternatives: dict[str, tuple[str, ...]] | None = None
    pending_ocr_document_kind: str | None = None
    pending_ocr_passport_role: str | None = None
    confirmed_documents: set[str] = field(default_factory=set)


class InMemoryDealStore:
    def __init__(
        self,
        ttl: timedelta,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sessions: dict[int, DealSession] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._generating: set[tuple[int, str, int]] = set()
        self._delete_callbacks: list[Callable[[int, str], None]] = []

    def add_delete_callback(self, callback: Callable[[int, str], None]) -> None:
        """Register synchronous cleanup for state owned by a deal session."""
        self._delete_callbacks.append(callback)

    def lock_for(self, user_id: int) -> asyncio.Lock:
        return self._locks.setdefault(user_id, asyncio.Lock())

    def create(self, user_id: int) -> DealSession:
        self.delete(user_id)
        now = self._clock()
        session = DealSession(
            deal_id=uuid4().hex[:12],
            owner_id=user_id,
            created_at=now,
            expires_at=now + self._ttl,
            form=ManualDealForm(),
        )
        self._sessions[user_id] = session
        return deepcopy(session)

    def get(self, user_id: int) -> DealSession | None:
        session = self._sessions.get(user_id)
        if session is not None and session.expires_at <= self._clock():
            self.delete(user_id)
            return None
        return deepcopy(session)

    async def submit_manual_field(
        self,
        user_id: int,
        deal_id: str,
        field_key: str,
        raw_value: str,
        expected_revision: int,
    ) -> DealSession:
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id:
                raise LookupError("Эта операция больше не активна.")
            if session.revision != expected_revision:
                raise LookupError("Получен устаревший ответ. Повторите ввод текущего поля.")
            if session.form is None:
                session.form = ManualDealForm()
            session.form.submit(field_key, raw_value)
            session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return deepcopy(session)

    async def finalize_manual_form(self, user_id: int, deal_id: str) -> DealDraft:
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id or session.form is None:
                raise LookupError("Эта операция больше не активна.")
            session.draft = session.form.build_deal()
            session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return session.draft.model_copy(deep=True)

    async def apply_confirmed_fields(
        self,
        user_id: int,
        deal_id: str,
        raw_values: dict[str, str],
        expected_revision: int,
    ) -> DealSession:
        """Apply only the OCR values the user explicitly confirmed."""
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id:
                raise LookupError("Эта операция больше не активна.")
            if session.revision != expected_revision:
                raise LookupError("Результат распознавания устарел.")
            if session.form is None:
                session.form = ManualDealForm()
            session.form.submit_many(raw_values)
            session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return deepcopy(session)

    async def stage_recognized_fields(
        self,
        user_id: int,
        deal_id: str,
        raw_values: dict[str, str],
        expected_revision: int,
        required_corrections: set[str] | None = None,
        alternatives: dict[str, tuple[str, ...]] | None = None,
        uncertain_fields: set[str] | None = None,
        document_kind: str | None = None,
        passport_role: str | None = None,
    ) -> DealSession:
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id or session.revision != expected_revision:
                raise LookupError("Эта операция больше не активна.")
            if session.form is None:
                session.form = ManualDealForm()
            corrections = set(required_corrections or ())
            # Validate trustworthy candidates without committing them. Values that
            # OCR marked for mandatory correction may intentionally have an invalid
            # format (for example a 16-character VIN); keep their raw text so the
            # user can copy and repair it instead of typing everything again.
            probe = ManualDealForm(values=deepcopy(session.form.values))
            for key, raw_value in raw_values.items():
                if key not in FIELD_BY_KEY:
                    raise ManualEntryError("Неизвестное поле договора.")
                if key not in corrections:
                    probe.submit(key, raw_value)
            session.pending_ocr_fields = dict(raw_values)
            for key, raw_value in raw_values.items():
                if key not in session.form.values:
                    continue
                candidate = ManualDealForm()
                try:
                    candidate.submit(key, raw_value)
                except ManualEntryError:
                    corrections.add(key)
                    continue
                if candidate.values[key] != session.form.values[key]:
                    corrections.add(key)
            session.pending_ocr_required_corrections = corrections
            session.pending_ocr_uncertain_fields = set(uncertain_fields or ()) | corrections
            session.pending_ocr_alternatives = deepcopy(alternatives or {})
            session.pending_ocr_document_kind = document_kind
            session.pending_ocr_passport_role = passport_role
            session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return deepcopy(session)

    async def confirm_recognized_fields(
        self, user_id: int, deal_id: str, expected_revision: int
    ) -> DealSession:
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id or session.revision != expected_revision:
                raise LookupError("Эта кнопка устарела.")
            if not session.pending_ocr_fields:
                raise LookupError("Нет данных для подтверждения.")
            if session.pending_ocr_required_corrections:
                raise ManualEntryError(
                    "Сначала исправьте поля, где найдено несколько разных значений."
                )
            if session.form is None:
                session.form = ManualDealForm()
            session.form.submit_many(session.pending_ocr_fields)
            if session.pending_ocr_document_kind == "passport":
                if session.pending_ocr_passport_role in {"seller", "buyer"}:
                    session.confirmed_documents.add(
                        f"passport:{session.pending_ocr_passport_role}"
                    )
            elif session.pending_ocr_document_kind:
                session.confirmed_documents.add(session.pending_ocr_document_kind)
            session.pending_ocr_fields = None
            session.pending_ocr_required_corrections = None
            session.pending_ocr_uncertain_fields = None
            session.pending_ocr_alternatives = None
            session.pending_ocr_document_kind = None
            session.pending_ocr_passport_role = None
            session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return deepcopy(session)

    async def update_pending_ocr_field(
        self,
        user_id: int,
        deal_id: str,
        field_key: str,
        raw_value: str,
        expected_revision: int,
    ) -> DealSession:
        """Validate and replace one not-yet-confirmed OCR value."""
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id or session.revision != expected_revision:
                raise LookupError("Эта кнопка устарела.")
            if not session.pending_ocr_fields or field_key not in session.pending_ocr_fields:
                raise LookupError("Поле больше не доступно для исправления.")
            probe = ManualDealForm()
            probe.submit(field_key, raw_value)
            session.pending_ocr_fields[field_key] = raw_value
            if session.pending_ocr_required_corrections is not None:
                session.pending_ocr_required_corrections.discard(field_key)
            if session.pending_ocr_uncertain_fields is not None:
                session.pending_ocr_uncertain_fields.discard(field_key)
            if session.pending_ocr_alternatives is not None:
                session.pending_ocr_alternatives.pop(field_key, None)
            session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return deepcopy(session)

    async def discard_pending_recognized_fields(
        self, user_id: int, deal_id: str, expected_revision: int
    ) -> DealSession:
        """Forget unconfirmed OCR values before retrying or switching to manual input."""
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.expires_at <= self._clock():
                self.delete(user_id)
                raise LookupError("Сделка не найдена или срок её хранения истёк.")
            if session.deal_id != deal_id or session.revision != expected_revision:
                raise LookupError("Эта кнопка устарела.")
            if session.pending_ocr_fields is not None:
                session.pending_ocr_fields = None
                session.pending_ocr_required_corrections = None
                session.pending_ocr_uncertain_fields = None
                session.pending_ocr_alternatives = None
                session.pending_ocr_document_kind = None
                session.pending_ocr_passport_role = None
                session.revision += 1
            session.expires_at = self._clock() + self._ttl
            return deepcopy(session)

    def delete(self, user_id: int) -> None:
        session = self._sessions.pop(user_id, None)
        if session is not None:
            self._generating = {
                marker
                for marker in self._generating
                if marker[:2] != (user_id, session.deal_id)
            }
            for callback in tuple(self._delete_callbacks):
                callback(user_id, session.deal_id)
        # Keep the per-user lock stable: removing a lock while a coroutine still
        # holds it could let a concurrent update enter through a newly created
        # lock. The allowlist bounds this small in-memory mapping.

    async def delete_if_current(
        self,
        user_id: int,
        deal_id: str,
        revision: int | None = None,
    ) -> bool:
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            if session is None or session.deal_id != deal_id:
                return False
            if revision is not None and session.revision != revision:
                return False
            self.delete(user_id)
            return True

    async def begin_generation(
        self, user_id: int, deal_id: str, revision: int
    ) -> bool:
        async with self.lock_for(user_id):
            session = self._sessions.get(user_id)
            marker = (user_id, deal_id, revision)
            if (
                session is None
                or session.deal_id != deal_id
                or session.revision != revision
                or session.draft is None
                or marker in self._generating
            ):
                return False
            self._generating.add(marker)
            return True

    async def finish_generation(
        self, user_id: int, deal_id: str, revision: int
    ) -> None:
        async with self.lock_for(user_id):
            self._generating.discard((user_id, deal_id, revision))

    def purge_expired(self) -> int:
        now = self._clock()
        expired = [uid for uid, session in self._sessions.items() if session.expires_at <= now]
        for uid in expired:
            self.delete(uid)
        return len(expired)

    async def purge_expired_async(self) -> int:
        removed = 0
        for user_id in tuple(self._sessions):
            async with self.lock_for(user_id):
                session = self._sessions.get(user_id)
                if session is not None and session.expires_at <= self._clock():
                    self.delete(user_id)
                    removed += 1
        return removed

    def clear(self) -> None:
        for user_id in tuple(self._sessions):
            self.delete(user_id)

