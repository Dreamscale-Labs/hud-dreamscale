"""Audit HUD leases using scoped provider billing, independent of trace costs."""

import time
from datetime import datetime
from uuid import UUID

FIELDS = (
    "id",
    "created_at",
    "terminated_at",
    "status",
    "hourly_rate",
    "api_key_id",
    "duration_hours",
    "cost",
)


def snapshot(api_key_id, client=None):
    UUID(api_key_id)
    if client is None:
        from hud.utils.platform import PlatformClient

        client = PlatformClient.from_settings()
    started = time.time()
    rows = {}
    for page in range(100):
        response = client.get(
            "/environments/usage",
            params={"days": 1, "api_key_id": api_key_id, "limit": 200, "offset": page * 200},
        )
        batch = response["environments"]
        for row in batch:
            if row["api_key_id"] != api_key_id or row["id"] in rows:
                raise ValueError("HUD usage pagination or key attribution changed; retry snapshot")
            rows[row["id"]] = {key: row.get(key) for key in FIELDS}
        if len(batch) < 200:
            return {
                "started_at": started,
                "finished_at": time.time(),
                "api_key_id": api_key_id,
                "environments": list(rows.values()),
            }
    raise ValueError("HUD usage exceeds bounded pagination")


def audit(before, after, expected_leases):
    if before["api_key_id"] != after["api_key_id"]:
        raise ValueError("HUD usage key changed")
    old = {row["id"] for row in before["environments"]}
    created = [row for row in after["environments"] if row["id"] not in old]
    preexisting_active = [row["id"] for row in before["environments"] if not row["terminated_at"]]
    in_interval = all(
        before["started_at"]
        <= datetime.fromisoformat(row["created_at"]).timestamp()
        <= after["finished_at"]
        for row in created
    )
    complete = (
        expected_leases > 0
        and len(created) == expected_leases
        and not preexisting_active
        and in_interval
        and all(row["status"] == "terminated" and row["terminated_at"] for row in created)
    )
    return {
        "cleanup_confirmed": bool(complete),
        "expected_leases": expected_leases,
        "observed_leases": len(created),
        "preexisting_active": preexisting_active,
        "provider_cost_usd": sum(float(row["cost"]) for row in created) if complete else None,
        "environment_ids": [row["id"] for row in created],
        "attribution": "Key-scoped before/after difference; billing IDs differ from runtime IDs",
    }


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-id", required=True, help="HUD key UUID, never the secret key")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--expected-leases", type=int)
    args = parser.parse_args()
    if args.before and (args.expected_leases is None or args.expected_leases <= 0):
        parser.error("--before requires a positive --expected-leases")
    result = snapshot(args.api_key_id)
    if args.before:
        result["audit"] = audit(json.loads(args.before.read_text()), result, args.expected_leases)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result.get("audit", {"environments": len(result["environments"])})))


if __name__ == "__main__":
    main()
