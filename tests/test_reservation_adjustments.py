"""End-to-end tests for reservation amount adjustments."""
from __future__ import annotations

import asyncio
import dataclasses
import os
import tempfile
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-adjustment-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "adjustments.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

import itertools

_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsadj-{name}-{next(_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


def _res(reservation_id: str, epsilon: float = 0.1,
         delta: float = 1e-7, ttl: int = 600) -> dict:
    return {
        "reservation_id": reservation_id,
        "epsilon": epsilon,
        "delta": delta,
        "ttl_seconds": ttl,
    }


def _adj(adjustment_id: str, epsilon: float = 0.2, delta: float = 1e-8) -> dict:
    return {"adjustment_id": adjustment_id, "epsilon": epsilon, "delta": delta}


def _ts(value: str) -> datetime:
    """Parse an ISO-8601 instant; GET views emit ``Z``, POST views ``+00:00``."""
    return datetime.fromisoformat(value)


async def _backdate(dataset_id: str, reservation_id: str,
                    expires_at: datetime) -> None:
    """Force a reservation's expiry instant (expiry is TTL based in production)."""
    async with db.connect() as conn:
        await db.backdate_reservation_expiry(
            conn,
            dataset_id=dataset_id,
            reservation_id=reservation_id,
            expires_at=expires_at.isoformat(),
        )
        await conn.commit()


def _projected(entries: list[tuple[float, float]], target_delta: float) -> float:
    """Independent composed-epsilon check via the rdp module."""
    _, epsilon, _ = rdp.evaluate_entries(entries, target_delta)
    return max(0.0, epsilon)


# ------------------------------------------------------------- happy path --


@pytest.mark.asyncio
async def test_adjustment_replaces_amount_and_budget_uses_it(client):
    ds = _new_dataset_id("basic")
    await _register(client, ds, eps=1.0, delta=1e-6)
    rv = await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-7)
    )
    assert rv.status_code == 201
    created = rv.json()

    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.35, delta=1e-8),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["idempotent_replay"] is False
    assert body["adjustment_id"] == "adj-1"
    assert body["dataset_id"] == ds
    assert body["reservation_id"] == "rv"
    assert body["allowed"] is True
    assert body["status"] == "adjusted"
    assert body["reason"] is None
    assert body["epsilon_before"] == pytest.approx(0.2)
    assert body["delta_before"] == pytest.approx(1e-7)
    assert body["epsilon_after"] == pytest.approx(0.35)
    assert body["delta_after"] == pytest.approx(1e-8)
    assert body["epsilon_cap"] == pytest.approx(1.0)
    # Single-reservation ledger: committed snapshots are the projection of
    # the old vs. the new amount alone.
    assert body["epsilon_committed_before"] == pytest.approx(
        _projected([(0.2, 1e-7)], 1e-6)
    )
    assert body["epsilon_committed_after"] == pytest.approx(
        _projected([(0.35, 1e-8)], 1e-6)
    )
    assert body["epsilon_remaining"] == pytest.approx(
        1.0 - body["epsilon_committed_after"]
    )

    # The reservation row now carries the new amount; TTL and expiry and the
    # creation-time admission snapshots are untouched.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.35)
    assert got["delta"] == pytest.approx(1e-8)
    assert got["ttl_seconds"] == created["ttl_seconds"]
    assert _ts(got["expires_at"]) == _ts(created["expires_at"])
    assert got["epsilon_before"] == pytest.approx(created["epsilon_before"])
    assert got["epsilon_after"] == pytest.approx(created["epsilon_after"])
    assert got["status"] == "granted"

    # Budget queries account with the new amount.
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["active_reservations"] == 1
    assert budget["epsilon_reserved"] == pytest.approx(
        _projected([(0.35, 1e-8)], 1e-6)
    )
    assert budget["epsilon_remaining"] == pytest.approx(
        1.0 - _projected([(0.35, 1e-8)], 1e-6)
    )


@pytest.mark.asyncio
async def test_chained_adjustments_move_amount_step_by_step(client):
    ds = _new_dataset_id("chain")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1))

    first = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2),
    )
    assert first.status_code == 201
    assert first.json()["epsilon_before"] == pytest.approx(0.1)

    second = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-2", epsilon=0.3),
    )
    assert second.status_code == 201
    assert second.json()["epsilon_before"] == pytest.approx(0.2)
    assert second.json()["epsilon_after"] == pytest.approx(0.3)
    assert second.json()["epsilon_committed_before"] == pytest.approx(
        first.json()["epsilon_committed_after"]
    )

    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_adjustment_downwards_frees_budget(client):
    ds = _new_dataset_id("down")
    await _register(client, ds, eps=0.5, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.4, delta=1e-7)
    )
    before = (await client.get(f"/datasets/{ds}/budget")).json()

    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.1, delta=1e-7),
    )
    assert r.status_code == 201
    after = (await client.get(f"/datasets/{ds}/budget")).json()
    assert after["epsilon_reserved"] < before["epsilon_reserved"]
    assert after["epsilon_remaining"] > before["epsilon_remaining"]
    assert after["epsilon_reserved"] == pytest.approx(
        _projected([(0.1, 1e-7)], 1e-6)
    )


@pytest.mark.asyncio
async def test_adjustment_to_pure_epsilon_uses_exact_addition(client):
    ds = _new_dataset_id("pure")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-7)
    )
    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2, delta=0.0),
    )
    assert r.status_code == 201
    body = r.json()
    # Pure epsilon-DP composes by exact addition: no RDP conversion overhead.
    assert body["epsilon_committed_after"] == pytest.approx(0.2)
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["epsilon_reserved"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_adjustment_accounts_requests_and_other_reservations(client):
    ds = _new_dataset_id("joint")
    await _register(client, ds, eps=1.0, delta=1e-6)
    # A granted, non-reversed request and another active reservation both
    # occupy budget alongside the adjusted reservation.
    await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-1", "query_type": "q",
              "epsilon": 0.2, "delta": 1e-8},
    )
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("other", epsilon=0.1, delta=1e-8)
    )
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1, delta=1e-8)
    )

    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.15, delta=1e-8),
    )
    assert r.status_code == 201
    body = r.json()
    expected_before = _projected(
        [(0.2, 1e-8), (0.1, 1e-8), (0.1, 1e-8)], 1e-6
    )
    expected_after = _projected(
        [(0.2, 1e-8), (0.1, 1e-8), (0.15, 1e-8)], 1e-6
    )
    assert body["epsilon_committed_before"] == pytest.approx(expected_before)
    assert body["epsilon_committed_after"] == pytest.approx(expected_after)
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["epsilon_remaining"] == pytest.approx(1.0 - expected_after)


# ------------------------------------------------------------- rejections --


@pytest.mark.asyncio
async def test_over_cap_adjustment_rejected_and_amount_kept(client):
    ds = _new_dataset_id("overcap")
    await _register(client, ds, eps=0.5, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-8)
    )
    budget_before = (await client.get(f"/datasets/{ds}/budget")).json()

    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.9, delta=1e-8),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["allowed"] is False
    assert body["status"] == "rejected"
    assert "budget exceeded" in body["reason"]
    assert body["epsilon_before"] == pytest.approx(0.2)
    assert body["epsilon_after"] == pytest.approx(0.9)
    # Occupancy snapshots: unchanged by the rejected attempt.
    assert body["epsilon_committed_after"] == pytest.approx(
        body["epsilon_committed_before"]
    )
    assert body["epsilon_committed_before"] == pytest.approx(
        _projected([(0.2, 1e-8)], 1e-6)
    )

    # The reservation keeps its old amount; the budget view is unchanged.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.2)
    assert got["delta"] == pytest.approx(1e-8)
    assert got["status"] == "granted"
    budget_after = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget_after["epsilon_reserved"] == pytest.approx(
        budget_before["epsilon_reserved"]
    )
    assert budget_after["epsilon_remaining"] == pytest.approx(
        budget_before["epsilon_remaining"]
    )

    # The rejected verdict is stored and replayed by the same adjustment_id.
    replay = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.9, delta=1e-8),
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["status"] == "rejected"
    assert replay.json()["created_at"] == body["created_at"]


@pytest.mark.asyncio
async def test_rejected_adjustment_does_not_block_later_legal_one(client):
    ds = _new_dataset_id("afterrej")
    await _register(client, ds, eps=0.5, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-8)
    )
    rejected = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.9, delta=1e-8),
    )
    assert rejected.json()["status"] == "rejected"

    ok = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-2", epsilon=0.3, delta=1e-8),
    )
    assert ok.status_code == 201
    assert ok.json()["status"] == "adjusted"
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.3)


# ------------------------------------------------------------ idempotency --


@pytest.mark.asyncio
async def test_adjustment_replay_returns_first_verdict(client):
    ds = _new_dataset_id("idem")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1))

    first = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.25, delta=1e-8),
    )
    assert first.status_code == 201

    for _ in range(3):
        again = await client.post(
            f"/datasets/{ds}/reservations/rv/adjustments",
            json=_adj("adj-1", epsilon=0.25, delta=1e-8),
        )
        assert again.status_code == 200
        body = again.json()
        assert body["idempotent_replay"] is True
        assert body == first.json() | {"idempotent_replay": True}

    # The amount moved exactly once.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_adjustment_replay_after_expiry_still_returns_record(client):
    ds = _new_dataset_id("idemexp")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv", ttl=10))
    first = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2),
    )
    assert first.status_code == 201

    # Force the reservation to expire, then retry the same adjustment: the
    # stored record is replayed instead of a 409 state conflict.
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    await _backdate(ds, "rv", past)
    replay = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2),
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["epsilon_after"] == pytest.approx(0.2)
    assert replay.json()["created_at"] == first.json()["created_at"]


@pytest.mark.asyncio
async def test_same_adjustment_id_different_payload_conflicts(client):
    ds = _new_dataset_id("idemconf")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv"))
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv2"))
    first = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2),
    )
    assert first.status_code == 201

    # Different epsilon on the same adjustment_id.
    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.21),
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjustment_id_conflict"

    # Different delta on the same adjustment_id.
    r = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2, delta=1e-9),
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjustment_id_conflict"

    # Same adjustment_id aimed at another reservation of this dataset.
    r = await client.post(
        f"/datasets/{ds}/reservations/rv2/adjustments",
        json=_adj("adj-1", epsilon=0.2),
    )
    assert r.status_code == 409
    assert r.json()["error"] == "adjustment_id_conflict"

    # None of the conflicts changed any amount.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.2)
    got2 = (await client.get(f"/datasets/{ds}/reservations/rv2")).json()
    assert got2["epsilon"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_adjustment_id_is_scoped_per_dataset(client):
    ds_a = _new_dataset_id("scope-a")
    ds_b = _new_dataset_id("scope-b")
    await _register(client, ds_a)
    await _register(client, ds_b)
    await client.post(f"/datasets/{ds_a}/reservations", json=_res("rv"))
    await client.post(f"/datasets/{ds_b}/reservations", json=_res("rv"))

    ra = await client.post(
        f"/datasets/{ds_a}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.2),
    )
    assert ra.status_code == 201
    # The same adjustment_id is an independent key in another dataset.
    rb = await client.post(
        f"/datasets/{ds_b}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.3),
    )
    assert rb.status_code == 201
    assert rb.json()["epsilon_after"] == pytest.approx(0.3)


# --------------------------------------------------------- state conflicts --


@pytest.mark.asyncio
async def test_adjustment_missing_dataset_or_reservation_404(client):
    ds = _new_dataset_id("notfound")
    r = await client.post(
        "/datasets/no-such-ds/reservations/rv/adjustments", json=_adj("adj-1")
    )
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"

    await _register(client, ds)
    r2 = await client.post(
        f"/datasets/{ds}/reservations/nope/adjustments", json=_adj("adj-1")
    )
    assert r2.status_code == 404
    assert r2.json()["error"] == "reservation_not_found"


@pytest.mark.asyncio
async def test_adjustment_terminal_states_conflict(client):
    ds = _new_dataset_id("states")
    await _register(client, ds, eps=1.0, delta=1e-6)

    # confirmed
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv-conf"))
    await client.post(
        f"/datasets/{ds}/reservations/rv-conf/confirm",
        json={"request_id": "req-1", "query_type": "q"},
    )
    r = await client.post(
        f"/datasets/{ds}/reservations/rv-conf/adjustments", json=_adj("adj-c")
    )
    assert r.status_code == 409
    assert r.json()["error"] == "reservation_confirmed"

    # released
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv-rel"))
    await client.post(f"/datasets/{ds}/reservations/rv-rel/release")
    r = await client.post(
        f"/datasets/{ds}/reservations/rv-rel/adjustments", json=_adj("adj-r")
    )
    assert r.status_code == 409
    assert r.json()["error"] == "reservation_released"

    # rejected
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv-rej", epsilon=5.0)
    )
    r = await client.post(
        f"/datasets/{ds}/reservations/rv-rej/adjustments", json=_adj("adj-j")
    )
    assert r.status_code == 409
    assert r.json()["error"] == "reservation_rejected"

    # expired (lazily swept inside the adjustment transaction)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv-exp", ttl=10)
    )
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    await _backdate(ds, "rv-exp", past)
    r = await client.post(
        f"/datasets/{ds}/reservations/rv-exp/adjustments", json=_adj("adj-e")
    )
    assert r.status_code == 409
    assert r.json()["error"] == "reservation_expired"
    got = (await client.get(f"/datasets/{ds}/reservations/rv-exp")).json()
    assert got["status"] == "expired"


# -------------------------------------------------------------- validation --


@pytest.mark.asyncio
async def test_adjustment_validation_errors(client):
    ds = _new_dataset_id("validation")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv"))

    for bad_eps in (0.0, -1.0, 1000.5):
        r = await client.post(
            f"/datasets/{ds}/reservations/rv/adjustments",
            json={"adjustment_id": "adj-x", "epsilon": bad_eps, "delta": 1e-8},
        )
        assert r.status_code == 422, (bad_eps, r.status_code, r.text)

    for bad_delta in (-1e-8, 1.0, 2.0):
        r = await client.post(
            f"/datasets/{ds}/reservations/rv/adjustments",
            json={"adjustment_id": "adj-x", "epsilon": 0.2, "delta": bad_delta},
        )
        assert r.status_code == 422, (bad_delta, r.status_code, r.text)

    bad_id = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json={"adjustment_id": "bad id!", "epsilon": 0.2, "delta": 1e-8},
    )
    assert bad_id.status_code == 422

    missing_id = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json={"epsilon": 0.2, "delta": 1e-8},
    )
    assert missing_id.status_code == 422

    missing_eps = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json={"adjustment_id": "adj-x", "delta": 1e-8},
    )
    assert missing_eps.status_code == 422

    missing_delta = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json={"adjustment_id": "adj-x", "epsilon": 0.2},
    )
    assert missing_delta.status_code == 422

    # Nothing was adjusted by any of the rejected payloads.
    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.1)
    assert got["delta"] == pytest.approx(1e-7)
    async with db.connect() as conn:
        row = await db.get_reservation_adjustment(conn, ds, "adj-x")
        assert row is None


# ------------------------------------------- interplay with existing flows --


@pytest.mark.asyncio
async def test_confirm_after_adjustment_uses_new_amount(client):
    ds = _new_dataset_id("confirm")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-8)
    )
    adj = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.3, delta=1e-7),
    )
    assert adj.status_code == 201

    verdict = await client.post(
        f"/datasets/{ds}/reservations/rv/confirm",
        json={"request_id": "req-1", "query_type": "q"},
    )
    assert verdict.status_code == 201
    body = verdict.json()
    assert body["allowed"] is True
    # The confirmed request carries the adjusted amount, and the budget is
    # not charged twice (reservation leaves the active set on confirm).
    assert body["epsilon"] == pytest.approx(0.3)
    assert body["delta"] == pytest.approx(1e-7)
    assert body["epsilon_after"] == pytest.approx(
        _projected([(0.3, 1e-7)], 1e-6)
    )
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["active_reservations"] == 0
    assert budget["granted_requests"] == 1
    assert budget["epsilon_consumed"] == pytest.approx(
        _projected([(0.3, 1e-7)], 1e-6)
    )


@pytest.mark.asyncio
async def test_release_forecast_uses_adjusted_amount(client):
    ds = _new_dataset_id("forecast")
    await _register(client, ds, eps=1.0, delta=1e-6)
    rv = await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-8)
    )
    expiry = _ts(rv.json()["expires_at"])

    adj = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.4, delta=1e-8),
    )
    assert adj.status_code == 201

    # Before expiry: the forecast's initial snapshot reserves the new amount.
    early = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    fc = await client.get(
        f"/datasets/{ds}/budget/release-forecast", params={"deadline": early}
    )
    assert fc.status_code == 200
    body = fc.json()
    assert body["initial"]["epsilon_reserved"] == pytest.approx(
        _projected([(0.4, 1e-8)], 1e-6)
    )
    assert body["initial"]["active_reservations"] == 1

    # Past expiry: the adjusted amount is what gets released.
    far = (expiry + timedelta(seconds=60)).isoformat()
    fc2 = await client.get(
        f"/datasets/{ds}/budget/release-forecast", params={"deadline": far}
    )
    body2 = fc2.json()
    assert len(body2["events"]) == 1
    assert body2["events"][0]["released_reservation_ids"] == ["rv"]
    assert body2["at_deadline"]["epsilon_reserved"] == pytest.approx(0.0)
    assert body2["at_deadline"]["epsilon_remaining"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_creation_replay_after_adjustment_uses_creation_payload(client):
    ds = _new_dataset_id("createreplay")
    await _register(client, ds, eps=1.0, delta=1e-6)
    created = await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-8)
    )
    assert created.status_code == 201

    adj = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.3, delta=1e-7),
    )
    assert adj.status_code == 201

    # Replaying the original creation payload still matches the creation-time
    # payload: idempotent replay, not a conflict, and no new reservation.
    replay = await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.2, delta=1e-8)
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    listing = (await client.get(f"/datasets/{ds}/reservations")).json()
    assert listing["count"] == 1

    # A payload equal to the *adjusted* amount is not the creation payload:
    # it conflicts like any other non-matching payload.
    conflict = await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.3, delta=1e-7)
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "reservation_id_conflict"


@pytest.mark.asyncio
async def test_adjustment_then_new_request_admission_uses_new_amount(client):
    ds = _new_dataset_id("admission")
    await _register(client, ds, eps=0.6, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1, delta=0.0)
    )
    adj = await client.post(
        f"/datasets/{ds}/reservations/rv/adjustments",
        json=_adj("adj-1", epsilon=0.5, delta=0.0),
    )
    assert adj.status_code == 201

    # Only 0.1 of pure-epsilon headroom remains after the adjustment.
    over = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-over", "query_type": "q",
              "epsilon": 0.2, "delta": 0.0},
    )
    assert over.json()["allowed"] is False
    fits = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-fits", "query_type": "q",
              "epsilon": 0.1, "delta": 0.0},
    )
    assert fits.json()["allowed"] is True


# ------------------------------------------------------------- concurrency --


@pytest.mark.asyncio
async def test_concurrent_duplicate_adjustments_apply_once(client):
    ds = _new_dataset_id("concdup")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1))

    results = await asyncio.gather(*[
        client.post(f"/datasets/{ds}/reservations/rv/adjustments",
                    json=_adj("adj-1", epsilon=0.25))
        for _ in range(20)
    ])
    created = [r for r in results if r.status_code == 201]
    replays = [r for r in results if r.status_code == 200]
    assert len(created) == 1
    assert len(replays) == 19

    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_concurrent_distinct_adjustments_all_recorded(client):
    ds = _new_dataset_id("concdist")
    await _register(client, ds)
    await client.post(f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1))

    n = 10
    results = await asyncio.gather(*[
        client.post(f"/datasets/{ds}/reservations/rv/adjustments",
                    json=_adj(f"adj-{i:02d}", epsilon=0.2))
        for i in range(n)
    ])
    assert all(r.status_code == 201 for r in results)

    got = (await client.get(f"/datasets/{ds}/reservations/rv")).json()
    assert got["epsilon"] == pytest.approx(0.2)

    async with db.connect() as conn:
        count = 0
        for i in range(n):
            row = await db.get_reservation_adjustment(conn, ds, f"adj-{i:02d}")
            assert row is not None
            count += 1
        assert count == n


@pytest.mark.asyncio
async def test_concurrent_adjustments_never_exceed_cap(client):
    ds = _new_dataset_id("conccap")
    await _register(client, ds, eps=0.5, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations", json=_res("rv", epsilon=0.1, delta=0.0)
    )

    # Each adjustment alone fits (0.4 <= 0.5); they all target the same
    # amount, so the cap can never be exceeded regardless of interleaving.
    results = await asyncio.gather(*[
        client.post(f"/datasets/{ds}/reservations/rv/adjustments",
                    json=_adj(f"adj-{i:02d}", epsilon=0.4, delta=0.0))
        for i in range(10)
    ])
    assert all(r.status_code == 201 for r in results)
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["epsilon_reserved"] == pytest.approx(0.4)
    assert budget["epsilon_remaining"] == pytest.approx(0.1)


# ------------------------------------------------------------- migration --


@pytest.mark.asyncio
async def test_migration_from_v6_schema(tmp_path, monkeypatch):
    old_db_path = str(tmp_path / "old.db")

    # Build a database at the pre-adjustment schema with some data.
    conn = await aiosqlite.connect(old_db_path)
    try:
        for schema in (
            db.SCHEMA_V1, db.SCHEMA_V2, db.SCHEMA_V3, db.SCHEMA_V4,
            db.SCHEMA_V5, db.SCHEMA_V6,
        ):
            await conn.executescript(schema)
        await conn.execute(
            "INSERT INTO datasets "
            "(dataset_id, epsilon_cap, delta_cap, description, created_at) "
            "VALUES ('legacy', 1.0, 1e-6, 'old', '2026-01-01T00:00:00+00:00')"
        )
        await conn.execute(
            "INSERT INTO reservations "
            "(dataset_id, reservation_id, epsilon, delta, ttl_seconds, status, "
            "reason, epsilon_before, epsilon_after, rdp_best_order, created_at, "
            "expires_at) VALUES ('legacy', 'rv1', 0.1, 1e-7, 600, 'granted', "
            "NULL, 0.0, 0.1, 1.1, '2026-01-01T00:00:01+00:00', "
            "'2026-01-01T00:10:01+00:00')"
        )
        await conn.execute("PRAGMA user_version = 6")
        await conn.commit()
    finally:
        await conn.close()

    patched = dataclasses.replace(settings, db_path=old_db_path)
    monkeypatch.setattr(db, "settings", patched)
    await db.init_db()

    async with db.connect() as c:
        async with c.execute("PRAGMA user_version") as cur:
            version = (await cur.fetchone())[0]
        assert version == db.LATEST_VERSION
        async with c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='reservation_adjustments'"
        ) as cur:
            assert await cur.fetchone() is not None
        row = await db.get_reservation(c, "legacy", "rv1")
        assert row["status"] == "granted"
        assert row["epsilon"] == pytest.approx(0.1)
        assert row["delta"] == pytest.approx(1e-7)
        # The creation payload columns are backfilled from the live amounts.
        assert row["created_epsilon"] == pytest.approx(0.1)
        assert row["created_delta"] == pytest.approx(1e-7)
