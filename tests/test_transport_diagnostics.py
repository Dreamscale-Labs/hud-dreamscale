"""Failure diagnostics retain operation/identity without changing replay policy."""

import json

import httpx
import pytest
from dropbear.errors import from_payload

from hud_dropbear.pooled import RequestFailedError, _error_fields, _http_operation
from tests.test_pooled import Server, predict


@pytest.mark.parametrize(
    "method,path,expected",
    [
        ("POST", "/v1/actions", "actions"),
        ("GET", "/v1/actions/request-id", "result_lookup"),
        ("POST", "/v1/actions/request-id/resolve", "resolve"),
        ("GET", "/v1/connection", "management"),
        ("DELETE", "/v1/inference/deployments/example", "management"),
    ],
)
async def test_response_hook_uses_bounded_route_labels(method, path, expected):
    events = []
    provider = Server().provider(emit=lambda name, **fields: events.append((name, fields)))
    request = httpx.Request(
        method,
        "https://do-not-record.example" + path + "?signed=SECRET_QUERY",
        headers={"Authorization": "Bearer SECRET_KEY"},
        extensions={"dropbear_request_id": "exact-request"},
    )
    try:
        await provider._response_hook(httpx.Response(200, request=request))
    finally:
        await provider.close()
    row = next(fields for name, fields in events if name == "provider_http_response")
    assert row["operation"] == expected == _http_operation(request)
    assert row["http_method"] == method and row["request_id"] == "exact-request"
    assert "SECRET" not in json.dumps(events)
    assert "do-not-record" not in json.dumps(events)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("gateway_overloaded", "gateway_overloaded"),
        ("SECRET key/path?", "unrecognized"),
        ("x" * 65, "unrecognized"),
    ],
)
def test_error_fields_exclude_messages_and_unbounded_codes(code, expected):
    error = from_payload(
        {"error": {"code": code, "problem": "SECRET response body", "cause": "https://SECRET/path"}}
    )
    result = _error_fields(error)
    assert result["error_type"] == "DropbearError"
    assert result["error_code"] == expected
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    "kind", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout]
)
async def test_original_transport_error_retained_without_an_inference_replay(kind):
    class TransportFailure(Server):
        async def handle(self, request):
            if request.url.path == "/v1/actions":
                self.posts.append(json.loads(request.content))
                raise kind("SECRET URL and exception details", request=request)
            return await super().handle(request)

    events, server = [], TransportFailure()
    async with server.provider(
        emit=lambda name, **fields: events.append((name, fields))
    ) as provider:
        with pytest.raises(RequestFailedError, match="request_not_admitted"):
            await predict(provider)
        assert provider.is_slot_available(0)
        assert provider._sequences[0] == 1
        assert not provider._pending
    post = next(fields for name, fields in events if name == "inference_post")
    assert post["error_type"] == kind.__name__ and post["error_code"] is None
    assert post["response_received"] is False
    assert len(server.posts) == len(server.resolutions) == 1
    assert server.posts[0]["request_id"] == server.resolutions[0]["request_id"]
    assert "SECRET" not in json.dumps(events)


async def test_recovery_rpc_errors_are_separate_from_original_post():
    class RecoveryFailures(Server):
        def __init__(self):
            super().__init__()
            self.mode = "unadmitted"
            self.lookup_calls = self.resolve_calls = 0

        async def handle(self, request):
            if request.method == "GET" and request.url.path.startswith("/v1/actions/"):
                self.lookup_calls += 1
                if self.lookup_calls == 1:
                    raise httpx.ConnectError("SECRET lookup route", request=request)
            if request.url.path.endswith("/resolve"):
                self.resolve_calls += 1
                if self.resolve_calls == 1:
                    raise httpx.ReadTimeout("SECRET resolve route", request=request)
            return await super().handle(request)

    events, server = [], RecoveryFailures()
    async with server.provider(
        emit=lambda name, **fields: events.append((name, fields))
    ) as provider:
        with pytest.raises(RequestFailedError):
            await predict(provider)
        assert provider.is_slot_available(0)
    errors = [fields for name, fields in events if name == "provider_http_error"]
    assert [(row["operation"], row["error_type"]) for row in errors] == [
        ("result_lookup", "ConnectError"),
        ("resolve", "ReadTimeout"),
    ]
    assert all(
        row["duration_s"] >= 0 and row["request_id"] == server.posts[0]["request_id"]
        for row in errors
    )
    assert len(server.posts) == 1 and len(server.resolutions) == 1
    assert "SECRET" not in json.dumps(events)


async def test_lost_success_recovers_result_without_replaying_actions():
    events, server = [], Server()
    server.mode = "response_lost"
    async with server.provider(
        emit=lambda name, **fields: events.append((name, fields))
    ) as provider:
        actions = await predict(provider)
        assert actions.shape == (10, 7)
    assert len(server.posts) == 1 and not server.resolutions
    responses = [
        fields
        for name, fields in events
        if name == "provider_http_response" and fields["request_id"]
    ]
    assert [(row["operation"], row["http_method"]) for row in responses] == [
        ("result_lookup", "GET")
    ]
    assert (
        next(fields for name, fields in events if name == "inference_post")["error_type"]
        == "ReadTimeout"
    )


async def test_http_api_rejection_is_not_mislabeled_as_transport_timeout():
    class Rejection(Server):
        async def handle(self, request):
            if request.url.path == "/v1/actions":
                self.posts.append(json.loads(request.content))
                return httpx.Response(
                    503, json={"error": {"code": "gateway_overloaded", "problem": "SECRET body"}}
                )
            return await super().handle(request)

    events, server = [], Rejection()
    async with server.provider(
        emit=lambda name, **fields: events.append((name, fields))
    ) as provider:
        with pytest.raises(RequestFailedError):
            await predict(provider)
    post = next(fields for name, fields in events if name == "inference_post")
    assert post["error_type"] == "DropbearError" and post["error_code"] == "gateway_overloaded"
    responses = [
        fields
        for name, fields in events
        if name == "provider_http_response" and fields["request_id"]
    ]
    assert [(row["operation"], row["http_status"]) for row in responses] == [
        ("actions", 503),
        ("result_lookup", 202),
        ("resolve", 422),
    ]
    assert len(server.posts) == 1 and "SECRET" not in json.dumps(events)
