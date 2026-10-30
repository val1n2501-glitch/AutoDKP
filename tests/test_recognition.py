from __future__ import annotations

import io
from datetime import timedelta
from types import SimpleNamespace

import pytest
from PIL import Image

from autodkp.domain.manual_entry import ManualEntryError
from autodkp.domain.models import DocumentKind
from autodkp.recognition.image_security import (
    UnsafeImageError,
    assess_image_quality,
    sanitize_image,
)
from autodkp.recognition.models import RecognizedDocument, RecognizedField
from autodkp.recognition.openai_provider import (
    _ALLOWED_FIELDS,
    OpenAIRecognitionProvider,
)
from autodkp.storage.session_store import InMemoryDealStore


def _jpeg_with_metadata() -> bytes:
    image = Image.new("RGB", (80, 40), "white")
    exif = Image.Exif()
    exif[0x010F] = "private-device-canary"
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif)
    return output.getvalue()


def test_sanitize_image_removes_metadata_and_reencodes() -> None:
    sanitized = sanitize_image(
        _jpeg_with_metadata(), max_bytes=1_000_000, max_pixels=1_000_000
    )
    assert b"private-device-canary" not in sanitized
    with Image.open(io.BytesIO(sanitized)) as image:
        assert image.format == "JPEG"
        assert image.getexif() == {}
        assert image.size == (80, 40)


def test_sanitize_image_rejects_non_image_and_oversized_input() -> None:
    with pytest.raises(UnsafeImageError, match="JPEG и PNG"):
        sanitize_image(b"not-an-image", max_bytes=100, max_pixels=1_000_000)
    with pytest.raises(UnsafeImageError, match="слишком большой"):
        sanitize_image(b"\xff\xd8\xff" + b"x" * 100, max_bytes=20, max_pixels=1_000_000)


def test_quality_check_warns_about_small_blank_photo() -> None:
    warnings_found = assess_image_quality(_jpeg_with_metadata())
    assert "низкое разрешение" in warnings_found
    assert "возможна сильная размытость" in warnings_found


def test_paper_pts_can_supply_current_sts_details() -> None:
    fields = set(_ALLOWED_FIELDS[DocumentKind.PAPER_PTS])
    assert {
        "vehicle.registration_plate",
        "vehicle.sts.series_number",
        "vehicle.sts.issued_by",
        "vehicle.sts.issue_date",
    } <= fields


class _FakeResponses:
    def __init__(self, outputs: list[RecognizedDocument]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.outputs.pop(0))


@pytest.mark.asyncio
async def test_paper_pts_uses_original_detail_and_rechecks_uncertain_critical_fields() -> None:
    first = RecognizedDocument(
        document_kind=DocumentKind.PAPER_PTS,
        fields=[
            RecognizedField(
                key="vehicle.make_model",
                value="LADA",
                confidence=0.96,
            ),
            RecognizedField(
                key="vehicle.vin", value="JMZGJ42683108161", confidence=0.96
            ),
        ],
    )
    second = RecognizedDocument(
        document_kind=DocumentKind.PAPER_PTS,
        fields=[
            RecognizedField(
                key="vehicle.make_model", value="MAZDA 6", confidence=0.98
            )
        ],
    )
    fake_responses = _FakeResponses([first, second])
    provider = OpenAIRecognitionProvider("sk-test-key-long-enough", "gpt-5.5")
    provider._client = SimpleNamespace(responses=fake_responses)

    result = await provider.recognize(DocumentKind.PAPER_PTS, [b"image"])

    assert len(fake_responses.calls) == 2
    assert all(
        call["input"][0]["content"][1]["detail"] == "original"
        for call in fake_responses.calls
    )
    assert all(call["reasoning"] == {"effort": "medium"} for call in fake_responses.calls)
    assert [
        item.value for item in result.fields if item.key == "vehicle.make_model"
    ] == ["LADA", "MAZDA 6"]


def test_brand_vin_conflict_marks_every_candidate_as_uncertain() -> None:
    document = RecognizedDocument(
        document_kind=DocumentKind.PAPER_PTS,
        fields=[
            RecognizedField(key="vehicle.make_model", value="LADA", confidence=0.9),
            RecognizedField(key="vehicle.make_model", value="LADA", confidence=0.95),
            RecognizedField(
                key="vehicle.vin", value="JMZGJ42683108161", confidence=0.99
            ),
        ],
    )
    OpenAIRecognitionProvider._flag_brand_vin_conflict(document)
    assert all(item.warning for item in document.fields)


@pytest.mark.asyncio
async def test_recognized_fields_require_confirmation() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"vehicle.vin": "XW7BF4FK90S123456"},
        session.revision,
    )
    assert staged.form is not None
    assert "vehicle.vin" not in staged.form.values
    confirmed = await store.confirm_recognized_fields(
        101, staged.deal_id, staged.revision
    )
    assert confirmed.form is not None
    assert confirmed.form.values["vehicle.vin"] == "XW7BF4FK90S123456"


@pytest.mark.asyncio
async def test_pending_recognized_field_can_be_corrected_before_confirmation() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"vehicle.registration_plate": "A123AA77"},
        session.revision,
    )
    corrected = await store.update_pending_ocr_field(
        101,
        staged.deal_id,
        "vehicle.registration_plate",
        "A123AA777",
        staged.revision,
    )
    assert corrected.pending_ocr_fields == {
        "vehicle.registration_plate": "A123AA777"
    }
    confirmed = await store.confirm_recognized_fields(
        101, corrected.deal_id, corrected.revision
    )
    assert confirmed.form is not None
    assert confirmed.form.values["vehicle.registration_plate"] == "A123AA777"


@pytest.mark.asyncio
async def test_conflicting_recognized_field_must_be_corrected_before_confirmation() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"vehicle.vin": "XW7BF4FK90S123456"},
        session.revision,
        {"vehicle.vin"},
        {
            "vehicle.vin": (
                "XW7BF4FK90S123456",
                "XW7BF4FK90S12345G",
            )
        },
    )
    assert staged.pending_ocr_alternatives == {
        "vehicle.vin": ("XW7BF4FK90S123456", "XW7BF4FK90S12345G")
    }
    assert staged.pending_ocr_uncertain_fields == {"vehicle.vin"}
    with pytest.raises(ManualEntryError, match="несколько разных"):
        await store.confirm_recognized_fields(101, staged.deal_id, staged.revision)

    corrected = await store.update_pending_ocr_field(
        101,
        staged.deal_id,
        "vehicle.vin",
        "XW7BF4FK90S123456",
        staged.revision,
    )
    assert not corrected.pending_ocr_required_corrections
    assert not corrected.pending_ocr_uncertain_fields
    assert not corrected.pending_ocr_alternatives
    confirmed = await store.confirm_recognized_fields(
        101, corrected.deal_id, corrected.revision
    )
    assert confirmed.form is not None
    assert confirmed.form.values["vehicle.vin"] == "XW7BF4FK90S123456"


@pytest.mark.asyncio
async def test_new_document_cannot_silently_overwrite_confirmed_value() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    first = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"vehicle.registration_plate": "A123AA77"},
        session.revision,
    )
    confirmed = await store.confirm_recognized_fields(
        101, first.deal_id, first.revision
    )
    second = await store.stage_recognized_fields(
        101,
        confirmed.deal_id,
        {"vehicle.registration_plate": "B456BB77"},
        confirmed.revision,
    )
    assert second.pending_ocr_required_corrections == {
        "vehicle.registration_plate"
    }
    with pytest.raises(ManualEntryError):
        await store.confirm_recognized_fields(101, second.deal_id, second.revision)
    current = store.get(101)
    assert current is not None and current.form is not None
    assert current.form.values["vehicle.registration_plate"] == "A123AA77"


@pytest.mark.asyncio
async def test_discard_pending_ocr_removes_unconfirmed_personal_data() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"seller.full_name": "Иванов Иван Иванович"},
        session.revision,
    )
    discarded = await store.discard_pending_recognized_fields(
        101, staged.deal_id, staged.revision
    )
    assert discarded.pending_ocr_fields is None
    assert discarded.pending_ocr_required_corrections is None
    assert discarded.pending_ocr_uncertain_fields is None
    assert discarded.pending_ocr_alternatives is None
    assert discarded.pending_ocr_document_kind is None
    assert discarded.pending_ocr_passport_role is None
    assert discarded.revision == staged.revision + 1


@pytest.mark.asyncio
async def test_pending_passport_fields_keep_buyer_or_seller_identity() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"buyer.full_name": "Иванов Иван Иванович"},
        session.revision,
        document_kind="passport",
        passport_role="buyer",
    )
    assert staged.pending_ocr_document_kind == "passport"
    assert staged.pending_ocr_passport_role == "buyer"
    assert set(staged.pending_ocr_fields or ()) == {"buyer.full_name"}
    confirmed = await store.confirm_recognized_fields(
        101, staged.deal_id, staged.revision
    )
    assert confirmed.confirmed_documents == {"passport:buyer"}


@pytest.mark.asyncio
async def test_correcting_one_uncertain_field_keeps_other_warning_markers() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {
            "vehicle.vin": "XW7BF4FK90S123456",
            "vehicle.engine_number": "30492167",
        },
        session.revision,
        uncertain_fields={"vehicle.vin", "vehicle.engine_number"},
    )
    corrected = await store.update_pending_ocr_field(
        101,
        staged.deal_id,
        "vehicle.vin",
        "XW7BF4FK90S123456",
        staged.revision,
    )
    assert corrected.pending_ocr_uncertain_fields == {"vehicle.engine_number"}


@pytest.mark.asyncio
async def test_invalid_ocr_vin_is_kept_for_user_correction() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"vehicle.vin": "JMZGJ42683108161"},
        session.revision,
        required_corrections={"vehicle.vin"},
        uncertain_fields={"vehicle.vin"},
        document_kind="paper_pts",
    )
    assert staged.pending_ocr_fields == {"vehicle.vin": "JMZGJ42683108161"}
    with pytest.raises(ManualEntryError, match="Сначала исправьте"):
        await store.confirm_recognized_fields(101, staged.deal_id, staged.revision)


@pytest.mark.asyncio
async def test_recognized_batch_is_atomic_and_stale_confirmation_fails() -> None:
    store = InMemoryDealStore(timedelta(minutes=60))
    session = store.create(101)
    with pytest.raises(ValueError):
        await store.stage_recognized_fields(
            101,
            session.deal_id,
            {
                "vehicle.vin": "XW7BF4FK90S123456",
                "vehicle.manufacture_year": "18",
            },
            session.revision,
        )
    current = store.get(101)
    assert current is not None and current.form is not None
    assert not current.form.values

    staged = await store.stage_recognized_fields(
        101,
        session.deal_id,
        {"vehicle.manufacture_year": "2020"},
        session.revision,
    )
    with pytest.raises(LookupError, match="устарела"):
        await store.confirm_recognized_fields(101, staged.deal_id, session.revision)
