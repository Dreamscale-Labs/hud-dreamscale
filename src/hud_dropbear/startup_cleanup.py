"""Exact-instance cleanup for an explicitly dedicated, otherwise idle HUD registry."""

import asyncio
from contextlib import asynccontextmanager


def _terminated(instance):
    return bool(instance.get("terminated_at")) or instance.get("status") in (
        "terminated",
        "stopped",
    )


async def verify_registry_selection(platform, registry_id, environment_name, *, build_id=None):
    """Bind name-based HUD placement to the registry we can clean up, before launch."""
    registry = await platform.aget(f"/registry/{registry_id}")
    if registry.get("id") != registry_id or registry.get("name") != environment_name:
        raise RuntimeError("Dedicated registry does not match the runtime environment name")
    if build_id and registry.get("latest_build_id") != build_id:
        raise RuntimeError("Dedicated registry latest build differs from the pinned build")
    return {
        "registry_id": registry_id,
        "environment_name": environment_name,
        "latest_build_id": registry.get("latest_build_id"),
    }


async def instance_inventory(platform, registry_id):
    inventory = {}
    offset = 0
    for _ in range(50):  # At most 10,000 rows; do not silently sample ownership.
        response = await platform.aget(
            "/instance",
            params={
                "registry_id": registry_id,
                "include_terminated": "true",
                "limit": 200,
                "offset": offset,
            },
        )
        instances = response.get("instances")
        if not isinstance(instances, list) or len(instances) > 200:
            raise RuntimeError("Cannot establish a complete, bounded instance inventory")
        for row in instances:
            if row.get("registry_id") != registry_id or not row.get("id"):
                raise RuntimeError("Instance inventory does not match the dedicated registry")
            if row["id"] in inventory:
                raise RuntimeError("Instance inventory changed during pagination")
            inventory[row["id"]] = row
        if not response.get("has_more"):
            return inventory
        if not instances:
            raise RuntimeError("Empty instance page declared more results")
        offset += len(instances)
    raise RuntimeError("Instance inventory exceeds the 10,000-row ownership bound")


async def verify_owned_instances(platform, registry_id, before_ids, *, build_id=None):
    """The caller guarantees exclusive registry use and an idle pre-launch snapshot.

    Ambiguous ownership fails closed. Never stop a preexisting instance or infer
    ownership from another user's shared registry. Polling is bounded by caller.
    """
    inventory = await instance_inventory(platform, registry_id)
    new_ids = set(inventory) - set(before_ids)
    if len(new_ids) != 1:
        return {
            "verified": False,
            "reason": "ambiguous_created_instances",
            "new_instance_ids": sorted(new_ids),
        }
    instance_id = new_ids.pop()
    row = inventory[instance_id]
    receipt = {"instance_id": instance_id, "build_id": row.get("build_id")}
    build_matches = not build_id or row.get("build_id") == build_id
    if not _terminated(row):
        stopped = await platform.apost(
            "/instance/stop",
            json={
                "registry_id": registry_id,
                "instance_ids": [instance_id],
            },
        )
        if stopped.get("errors"):
            return {**receipt, "verified": False, "reason": "stop_not_acknowledged"}
    for _ in range(30):
        inventory = await instance_inventory(platform, registry_id)
        row = inventory.get(instance_id)
        if row is not None and _terminated(row):
            return {
                **receipt,
                "verified": bool(build_matches),
                "termination_confirmed": True,
                "reason": "instance_terminated" if build_matches else "build_changed",
            }
        await asyncio.sleep(2)
    return {**receipt, "verified": False, "reason": "termination_not_observed"}


class ExclusiveRegistryCampaign:
    """Bound a concurrent runtime pool in a dedicated registry, including failed starts."""

    def __init__(
        self,
        platform,
        registry_id,
        *,
        environment_name,
        expected_instances,
        build_id=None,
        retry_allowance=0,
    ):
        if type(expected_instances) is not int or not 1 <= expected_instances <= 64:
            raise ValueError("Campaign must own between one and 64 instances")
        if type(retry_allowance) is not int or retry_allowance < 0:
            raise ValueError("retry_allowance must be a non-negative integer")
        self.platform, self.registry_id = platform, registry_id
        self.environment_name = environment_name
        self.expected_instances, self.build_id = expected_instances, build_id
        # Lane readiness retries release an unready simulator before leasing another,
        # so the campaign may create (and still owns) that many extra terminated rows.
        self.retry_allowance = retry_allowance
        self.before_ids = None
        self.owned_ids = None
        self.cleanup_receipt = {"verified": False, "reason": "not_started"}

    async def __aenter__(self):
        self.registry_receipt = await verify_registry_selection(
            self.platform, self.registry_id, self.environment_name, build_id=self.build_id
        )
        inventory = await instance_inventory(self.platform, self.registry_id)
        if any(not _terminated(row) for row in inventory.values()):
            raise RuntimeError("Dedicated campaign registry already has an active instance")
        self.before_ids = set(inventory)
        return self

    async def _created(self, *, validate_build=True, enforce_live=True):
        inventory = await instance_inventory(self.platform, self.registry_id)
        created = {key: row for key, row in inventory.items() if key not in self.before_ids}
        if len(created) > self.expected_instances + self.retry_allowance:
            raise RuntimeError("Created instance count exceeds exclusive campaign ownership")
        live = [key for key, row in created.items() if not _terminated(row)]
        if enforce_live and len(live) > self.expected_instances:
            # Within the created bound every row is ours; cleanup still stops them all.
            raise RuntimeError("Live created instance count exceeds exclusive campaign ownership")
        if (
            validate_build
            and self.build_id
            and any(row.get("build_id") != self.build_id for row in created.values())
        ):
            raise RuntimeError("Campaign instance build differs from the pinned build")
        return created

    async def validate_ready(self):
        created = await self._created()
        live = sorted(key for key, row in created.items() if not _terminated(row))
        if len(live) != self.expected_instances:
            raise RuntimeError("Campaign has not established every expected live instance")
        # Released retry simulators remain owned so cleanup verifies their termination.
        self.owned_ids = set(created)
        return {
            **self.registry_receipt,
            "instance_ids": live,
            "retired_instance_ids": sorted(set(created) - set(live)),
            "build_ids": sorted({str(row.get("build_id")) for row in created.values()}),
        }

    async def _cleanup(self):
        # A wrong build invalidates the run, not ownership of its allocated instances.
        if self.owned_ids is None:
            created = await self._created(validate_build=False, enforce_live=False)
            owned = set(created)
        else:
            # Once verified, these exact IDs remain ours even if another runtime
            # appears in the registry. Fail the campaign, but still stop our IDs.
            inventory = await instance_inventory(self.platform, self.registry_id)
            created = {key: row for key, row in inventory.items() if key not in self.before_ids}
            owned = self.owned_ids
        changed = set(created) != owned
        if not owned:
            return {"verified": False, "reason": "no_created_instance_evidence"}
        live = [key for key in owned if key not in created or not _terminated(created[key])]
        if live:
            result = await self.platform.apost(
                "/instance/stop",
                json={
                    "registry_id": self.registry_id,
                    "instance_ids": sorted(live),
                },
            )
            if result.get("errors"):
                return {
                    "verified": False,
                    "reason": "stop_not_acknowledged",
                    "instance_ids": sorted(owned),
                }
        for _ in range(30):
            inventory = await instance_inventory(self.platform, self.registry_id)
            if all(key in inventory and _terminated(inventory[key]) for key in owned):
                return {
                    "verified": not changed,
                    "reason": "owned_instance_inventory_changed"
                    if changed
                    else "instances_terminated",
                    "termination_confirmed": True,
                    "instance_ids": sorted(owned),
                    "unexpected_instance_ids": sorted(set(created) - owned),
                }
            await asyncio.sleep(2)
        return {
            "verified": False,
            "reason": "termination_not_observed",
            "instance_ids": sorted(owned),
        }

    async def __aexit__(self, exc_type, exc, traceback):
        cleanup = asyncio.create_task(asyncio.wait_for(self._cleanup(), 120))
        cancelled = False
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            self.cleanup_receipt = cleanup.result()
        except Exception as error:
            self.cleanup_receipt = {"verified": False, "reason": type(error).__name__}
            if exc is not None:
                exc.add_note(f"Campaign cleanup unresolved: {type(error).__name__}")
                return False
            raise
        if cancelled:
            raise asyncio.CancelledError
        if not self.cleanup_receipt["verified"]:
            if exc is not None:
                exc.add_note(f"Campaign cleanup unresolved: {self.cleanup_receipt['reason']}")
                return False
            raise RuntimeError(f"Campaign cleanup is unresolved: {self.cleanup_receipt['reason']}")


class ExclusiveRegistryRuntime:
    """Wrap HUDRuntime with read-only before inventory and explicit after cleanup."""

    def __init__(self, provider, platform, registry_id, *, build_id=None):
        self.provider, self.platform, self.registry_id = provider, platform, registry_id
        self.build_id = build_id
        self.before_ids = None

    @asynccontextmanager
    async def __call__(self, task):
        # Sequential startup profiling only; every trial must finish verification.
        if self.before_ids is not None:
            raise RuntimeError("Previous runtime cleanup has not been verified")
        await verify_registry_selection(
            self.platform, self.registry_id, task.env, build_id=self.build_id
        )
        inventory = await instance_inventory(self.platform, self.registry_id)
        if any(not _terminated(row) for row in inventory.values()):
            raise RuntimeError("Dedicated startup registry already has an active instance")
        self.before_ids = set(inventory)
        async with self.provider(task) as runtime:
            yield runtime

    async def verify_cleanup(self, runtime):
        if self.before_ids is None:
            return {"verified": False, "reason": "no_prelaunch_inventory"}
        receipt = await verify_owned_instances(
            self.platform, self.registry_id, self.before_ids, build_id=self.build_id
        )
        if receipt["verified"]:
            self.before_ids = None
        return receipt
