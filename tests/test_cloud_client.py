"""Tests for the openflight-cloud HTTP client (wire contract)."""

import json

import pytest

from openflight.cloud import client as cl


class FakeTransport:
    """Records requests and returns queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, method, url, data=None, headers=None, timeout=30):
        self.calls.append({"method": method, "url": url, "data": data, "headers": headers or {}})
        return self._responses.pop(0)


def _resp(status, body=None, headers=None):
    raw = json.dumps(body).encode() if body is not None else b""
    return cl.HttpResponse(status=status, headers=headers or {}, body=raw)


class TestHealth:
    def test_ok_returns_true(self):
        client = cl.CloudClient(
            "https://e.test", request_fn=FakeTransport([_resp(200, {"status": "ok"})])
        )
        assert client.health() is True

    def test_non_ok_returns_false(self):
        client = cl.CloudClient("https://e.test", request_fn=FakeTransport([_resp(503)]))
        assert client.health() is False

    def test_network_error_returns_false(self):
        def boom(*a, **k):
            raise cl.CloudNetworkError("offline")

        client = cl.CloudClient("https://e.test", request_fn=boom)
        assert client.health() is False

    def test_uses_v1_health_path(self):
        t = FakeTransport([_resp(200, {"status": "ok"})])
        cl.CloudClient("https://e.test/", request_fn=t).health()
        assert t.calls[0]["url"] == "https://e.test/v1/health"
        assert t.calls[0]["method"] == "GET"


class TestDeviceLinkStart:
    def test_parses_link_start_response(self):
        body = {
            "link_code": "ABCD-2345",
            "poll_token": "secret-token",
            "interval_s": 5,
            "expires_s": 900,
        }
        t = FakeTransport([_resp(200, body)])
        client = cl.CloudClient("https://e.test", request_fn=t)
        result = client.device_link_start("garage pi", "0.2.0")
        assert result.link_code == "ABCD-2345"
        assert result.poll_token == "secret-token"
        assert result.interval_s == 5
        assert result.expires_s == 900

    def test_sends_device_name_and_version(self):
        t = FakeTransport(
            [_resp(200, {"link_code": "A", "poll_token": "p", "interval_s": 5, "expires_s": 900})]
        )
        cl.CloudClient("https://e.test", request_fn=t).device_link_start("garage pi", "9.9.9")
        sent = json.loads(t.calls[0]["data"].decode())
        assert sent == {"device_name": "garage pi", "client_version": "9.9.9"}
        assert t.calls[0]["url"] == "https://e.test/v1/device-link/start"

    def test_422_raises_link_error(self):
        t = FakeTransport([_resp(422, {"reason": "invalid_device_name"})])
        with pytest.raises(cl.LinkError):
            cl.CloudClient("https://e.test", request_fn=t).device_link_start("", "0.2.0")

    def test_429_raises_rate_limited_with_retry_after(self):
        t = FakeTransport([_resp(429, {"reason": "rate_limited"}, headers={"Retry-After": "42"})])
        with pytest.raises(cl.RateLimited) as exc:
            cl.CloudClient("https://e.test", request_fn=t).device_link_start("pi", "0.2.0")
        assert exc.value.retry_after == 42


class TestDeviceLinkPoll:
    def test_pending(self):
        t = FakeTransport([_resp(200, {"status": "pending"})])
        result = cl.CloudClient("https://e.test", request_fn=t).device_link_poll("tok")
        assert result.status == "pending"

    def test_linked_returns_token_and_id(self):
        body = {"status": "linked", "device_token": "of_device_x", "device_id": "uuid-1"}
        t = FakeTransport([_resp(200, body)])
        result = cl.CloudClient("https://e.test", request_fn=t).device_link_poll("tok")
        assert result.status == "linked"
        assert result.device_token == "of_device_x"
        assert result.device_id == "uuid-1"

    def test_404_maps_to_unknown(self):
        t = FakeTransport([_resp(404, {"reason": "unknown_poll_token"})])
        result = cl.CloudClient("https://e.test", request_fn=t).device_link_poll("tok")
        assert result.status == "unknown"

    def test_429_raises_rate_limited(self):
        t = FakeTransport([_resp(429, {}, headers={"Retry-After": "5"})])
        with pytest.raises(cl.RateLimited):
            cl.CloudClient("https://e.test", request_fn=t).device_link_poll("tok")


class TestUploadSession:
    def _client(self, responses):
        return cl.CloudClient(
            "https://e.test", token="of_device_tok", request_fn=FakeTransport(responses)
        )

    def test_201_is_success(self):
        r = self._client([_resp(201, {"session_id": "s1", "shot_count": 7})]).upload_session(
            "s1", b"gz"
        )
        assert r.action == "success"
        assert r.shot_count == 7

    def test_200_is_success(self):
        r = self._client([_resp(200, {"session_id": "s1", "shot_count": 7})]).upload_session(
            "s1", b"gz"
        )
        assert r.action == "success"

    def test_401_needs_relink(self):
        r = self._client([_resp(401, {"reason": "invalid_or_revoked_token"})]).upload_session(
            "s1", b"gz"
        )
        assert r.action == "relink"

    def test_402_quota(self):
        r = self._client([_resp(402, {"reason": "quota_exceeded"})]).upload_session("s1", b"gz")
        assert r.action == "quota"

    def test_413_parks(self):
        r = self._client([_resp(413, {"reason": "body_too_large"})]).upload_session("s1", b"gz")
        assert r.action == "park"

    def test_422_parks(self):
        r = self._client([_resp(422, {"reason": "invalid_gzip"})]).upload_session("s1", b"gz")
        assert r.action == "park"
        assert r.reason == "invalid_gzip"

    def test_429_retry_with_retry_after(self):
        r = self._client(
            [_resp(429, {"reason": "rate_limited"}, headers={"Retry-After": "30"})]
        ).upload_session("s1", b"gz")
        assert r.action == "rate_limited"
        assert r.retry_after == 30

    def test_5xx_retry(self):
        r = self._client([_resp(503)]).upload_session("s1", b"gz")
        assert r.action == "retry"

    def test_sends_bearer_auth_and_gzip_headers(self):
        t = FakeTransport([_resp(201, {"session_id": "s1", "shot_count": 1})])
        cl.CloudClient("https://e.test", token="of_device_tok", request_fn=t).upload_session(
            "1f0e9c2a-7b3d-4e5f-8a9b-0c1d2e3f4a5b", b"gzbytes"
        )
        call = t.calls[0]
        assert call["method"] == "PUT"
        assert call["url"] == "https://e.test/v1/sessions/1f0e9c2a-7b3d-4e5f-8a9b-0c1d2e3f4a5b"
        assert call["headers"]["Authorization"] == "Bearer of_device_tok"
        assert call["headers"]["Content-Type"] == "application/x-ndjson"
        assert call["headers"]["Content-Encoding"] == "gzip"
        assert call["data"] == b"gzbytes"
