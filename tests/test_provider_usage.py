import pytest

from hud_dreamscale.provider_usage import audit, snapshot

KEY = "366ca626-2032-4af9-a664-812678f58098"


def lease(identity="one"):
    return dict(
        id=identity,
        api_key_id=KEY,
        created_at="2026-09-27T00:00:00+00:00",
        terminated_at="2026-09-27T00:01:00+00:00",
        status="terminated",
        cost=0.008333,
    )


def samples(rows):
    return (
        {"api_key_id": KEY, "environments": [], "started_at": 0},
        {"api_key_id": KEY, "environments": rows, "finished_at": 2000000000},
    )


def test_usage_requires_exact_lease_count_and_termination():
    before, after = samples([lease()])
    assert audit(before, after, 1)["cleanup_confirmed"]
    assert audit(before, after, 1)["provider_cost_usd"] == 0.008333
    assert not audit(before, after, 2)["cleanup_confirmed"]
    after["environments"][0]["terminated_at"] = None
    assert not audit(before, after, 1)["cleanup_confirmed"]


def test_preexisting_active_or_outside_interval_does_not_confirm_cleanup():
    before, after = samples([lease()])
    before["environments"] = [dict(lease("other"), terminated_at=None)]
    assert not audit(before, after, 1)["cleanup_confirmed"]
    before["environments"] = []
    before["started_at"] = 2000000000
    assert not audit(before, after, 1)["cleanup_confirmed"]


def test_paginate_without_exposing_names():
    class Client:
        def get(self, path, *, params):
            assert path == "/environments/usage" and params["api_key_id"] == KEY
            rows = [dict(lease(str(i)), user_name="private") for i in range(201)]
            return {"environments": rows[params["offset"] : params["offset"] + 200]}

    result = snapshot(KEY, Client())
    assert len(result["environments"]) == 201
    assert all("user_name" not in row for row in result["environments"])


def test_wrong_key_is_not_attributed():
    class Client:
        def get(self, path, *, params):
            return {"environments": [dict(lease(), api_key_id="different")]}

    with pytest.raises(ValueError, match="attribution"):
        snapshot(KEY, Client())


def test_unrelated_manual_environment_must_be_declared_in_baseline():
    before, after = samples([lease()])
    before["environments"] = [dict(lease("manual"), terminated_at=None)]
    before["unrelated_active_environment_ids"] = ["manual"]
    assert audit(before, after, 1)["cleanup_confirmed"]
    before["unrelated_active_environment_ids"] = ["one"]
    with pytest.raises(ValueError, match="active baseline"):
        audit(before, after, 1)
