from collections import Counter
from uuid import uuid4

import httpx
import pytest

from hud_dreamscale.runtime_cleanup import delete_owned_sessions


async def test_delete_retries_rejections_but_never_claims_provider_termination():
    identities = [str(uuid4()), str(uuid4())]
    calls = Counter()

    def handler(request):
        identity = request.url.path.rsplit("/", 1)[1]
        assert identity in identities
        calls[identity] += 1
        return httpx.Response(429 if calls[identity] == 1 else 204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await delete_owned_sessions(
            identities, api_key="test", runtime_url="https://mcp.hud.ai", client=client
        )
    assert result["delete_acknowledged"] and not result["provider_termination_confirmed"]
    assert calls == {identity: 2 for identity in identities}


async def test_wrong_origin_or_duplicate_identity_fails_before_network():
    identity = str(uuid4())
    for ids, origin in [
        ([identity], "http://mcp.hud.ai"),
        ([identity, identity], "https://mcp.hud.ai"),
    ]:
        with pytest.raises(ValueError):
            await delete_owned_sessions(ids, api_key="test", runtime_url=origin)
