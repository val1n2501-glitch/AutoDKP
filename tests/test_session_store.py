from datetime import UTC, datetime, timedelta

import pytest

from autodkp.domain.demo import make_demo_deal
from autodkp.domain.manual_entry import MANUAL_FIELDS
from autodkp.storage.session_store import InMemoryDealStore


def _demo_answers() -> dict[str, str]:
    deal = make_demo_deal()
    values = deal.model_dump()

    def lookup(key: str) -> object:
        current: object = values
        for part in key.split("."):
            assert isinstance(current, dict)
            current = current[part]
        return current

    answers = {}
    for spec in MANUAL_FIELDS:
        value = lookup(spec.key)
        answers[spec.key] = value.strftime("%d.%m.%Y") if hasattr(value, "strftime") else str(value)
    return answers


def test_one_active_session_per_user() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    first = store.create(101)
    second = store.create(101)
    assert first.deal_id != second.deal_id
    assert store.get(101) == second


def test_sessions_are_isolated_by_user() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    alice = store.create(101)
    bob = store.create(202)
    assert alice.owner_id == 101
    assert bob.owner_id == 202
    assert alice.deal_id != bob.deal_id


def test_returned_session_is_a_copy() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    store.create(101)
    snapshot = store.get(101)
    assert snapshot is not None and snapshot.form is not None
    snapshot.form.values["agreement_place"] = "не должно сохраниться"
    fresh = store.get(101)
    assert fresh is not None and fresh.form is not None
    assert "agreement_place" not in fresh.form.values


def test_user_lock_remains_stable_across_delete_and_recreate() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    original_lock = store.lock_for(101)
    store.create(101)
    store.delete(101)
    store.create(101)
    assert store.lock_for(101) is original_lock


@pytest.mark.asyncio
async def test_manual_answers_are_isolated_and_can_be_finalized() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    session = store.create(101)
    for spec in MANUAL_FIELDS:
        current = store.get(101)
        assert current is not None
        await store.submit_manual_field(
            101,
            session.deal_id,
            spec.key,
            _demo_answers()[spec.key],
            current.revision,
        )
    result = await store.finalize_manual_form(101, session.deal_id)
    assert result == make_demo_deal()
    current = store.get(101)
    assert current is not None
    assert await store.begin_generation(101, current.deal_id, current.revision)
    assert not await store.begin_generation(101, current.deal_id, current.revision)
    await store.finish_generation(101, current.deal_id, current.revision)
    assert await store.begin_generation(101, current.deal_id, current.revision)


@pytest.mark.asyncio
async def test_stale_deal_id_cannot_update_replacement_session() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    stale = store.create(101)
    store.create(101)
    with pytest.raises(LookupError, match="не активна"):
        await store.submit_manual_field(
            101, stale.deal_id, "agreement_place", "г. Москва", stale.revision
        )


@pytest.mark.asyncio
async def test_concurrent_answer_with_stale_revision_is_rejected() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    session = store.create(101)
    await store.submit_manual_field(
        101, session.deal_id, "agreement_place", "г. Москва", session.revision
    )
    with pytest.raises(LookupError, match="устаревший ответ"):
        await store.submit_manual_field(
            101, session.deal_id, "agreement_place", "г. Тверь", session.revision
        )
    current = store.get(101)
    assert current is not None and current.form is not None
    assert current.form.values["agreement_place"] == "г. Москва"


@pytest.mark.asyncio
async def test_stale_delete_cannot_remove_new_deal() -> None:
    store = InMemoryDealStore(ttl=timedelta(minutes=60))
    old = store.create(101)
    new = store.create(101)
    assert not await store.delete_if_current(101, old.deal_id, old.revision)
    assert store.get(101) == new


@pytest.mark.asyncio
async def test_expired_session_is_purged_without_user_lookup() -> None:
    current_time = datetime(2026, 10, 21, tzinfo=UTC)

    def clock() -> datetime:
        return current_time

    store = InMemoryDealStore(ttl=timedelta(minutes=5), clock=clock)
    deleted: list[tuple[int, str]] = []
    store.add_delete_callback(lambda user_id, deal_id: deleted.append((user_id, deal_id)))
    session = store.create(101)
    current_time += timedelta(minutes=6)
    assert await store.purge_expired_async() == 1
    assert not store._sessions
    assert deleted == [(101, session.deal_id)]
