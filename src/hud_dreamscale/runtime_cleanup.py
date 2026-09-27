"""Retry DELETE only for recorded, owned HUD runtime sessions.

An acknowledgement is not provider termination evidence. Billing reconciliation
remains mandatory because HUD session IDs and billed environment IDs differ.
"""

import asyncio
import time
from uuid import UUID

import httpx


async def delete_owned_sessions(session_ids, *, api_key, runtime_url, client=None):
    if runtime_url.rstrip("/") != "https://mcp.hud.ai":
        raise ValueError("production HUD runtime origin required")
    identities = list(session_ids)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate runtime session identity")
    for identity in identities:
        UUID(identity)
    if client is None:
        async with httpx.AsyncClient(timeout=10) as owned:
            return await delete_owned_sessions(
                identities, api_key=api_key, runtime_url=runtime_url, client=owned
            )
    gate = asyncio.Semaphore(8)
    results = {}

    async def one(identity):
        async with gate:
            for attempt in range(3):
                try:
                    response = await client.delete(
                        f"{runtime_url.rstrip('/')}/runtime/sessions/{identity}",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    code = response.status_code
                except httpx.HTTPError:
                    code = "transport_error"
                results[identity] = {
                    "session_id": identity,
                    "status": code,
                    "attempts": attempt + 1,
                }
                if code in (200, 204, 404):
                    return
                if attempt < 2:
                    await asyncio.sleep(attempt + 1)

    started = time.monotonic()
    timed_out = False
    try:
        async with asyncio.timeout(90):
            await asyncio.gather(*(one(identity) for identity in identities))
    except TimeoutError:
        timed_out = True
    return {
        "duration_s": time.monotonic() - started,
        "timed_out": timed_out,
        "delete_acknowledged": not timed_out
        and len(results) == len(identities)
        and all(x["status"] in (200, 204, 404) for x in results.values()),
        "rows": list(results.values()),
        "provider_termination_confirmed": False,
    }
