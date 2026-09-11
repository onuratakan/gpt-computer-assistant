"""Behavior tests: mocked HTTP transport; no endpoint probes or paid calls."""

from __future__ import annotations

import asyncio
import copy
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from upsonic.tools.custom_tools import agent_guild as ag

_REAL_CLIENT = httpx.Client
_REAL_ASYNC_CLIENT = httpx.AsyncClient
_FIXTURE = Path(__file__).parent / "fixtures" / "agent_guild_preflight.json"
_TARGET = "https://agent-guild-5d5r.onrender.com/mcp"


class Stream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, body: bytes, delay: float = 0) -> None:
        self.body = body
        self.delay = delay
        self.closed = False

    def __iter__(self):
        yield self.body

    async def __aiter__(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        yield self.body

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class AgentGuildBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = json.loads(_FIXTURE.read_text())
        self.requests: list[httpx.Request] = []
        self.options: list[dict[str, object]] = []
        self.streams: list[Stream] = []
        self.body = _FIXTURE.read_bytes()
        self.status = 200
        self.headers = {"Content-Type": "application/json"}
        self.delay = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        stream = Stream(self.body, self.delay)
        self.streams.append(stream)
        return httpx.Response(self.status, headers=self.headers, stream=stream)

    def client(self, **kwargs) -> httpx.Client:
        self.options.append(kwargs)
        return _REAL_CLIENT(transport=httpx.MockTransport(self.handler), **kwargs)

    def async_client(self, **kwargs) -> httpx.AsyncClient:
        self.options.append(kwargs)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(self.handler), **kwargs)

    def run_observation(self, value: dict | None = None, asynchronous: bool = False) -> dict:
        if value is not None:
            self.body = json.dumps(value).encode()
        toolkit = ag.AgentGuildTools([_TARGET], use_async=asynchronous)
        with patch.object(ag.httpx, "Client", side_effect=self.client), patch.object(
            ag.httpx, "AsyncClient", side_effect=self.async_client
        ):
            text = asyncio.run(toolkit.aobserve_endpoint(_TARGET)) if asynchronous else toolkit.observe_endpoint(_TARGET)
        return json.loads(text)

    def test_current_retained_fixture_preserves_failed_and_unknown_checks(self) -> None:
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                result = self.run_observation(asynchronous=asynchronous)
                self.assertEqual(result["status"], "observed")
                self.assertEqual(result["failed"], ["agent_card_signed"])
                self.assertEqual(result["unknowns"], ["payment_claim_holds", "independent_evidence"])
                self.assertEqual(result["checks"], [{"check": x["check"], "status": x["status"]} for x in self.observation["checks"]])
                self.assertNotIn("verdict", result)

    def test_transmits_only_fixed_service_and_exact_selected_url(self) -> None:
        for asynchronous in (False, True):
            self.run_observation(asynchronous=asynchronous)
        for request in self.requests:
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url.copy_with(query=None)), ag._SERVICE)
            self.assertEqual(list(request.url.params.multi_items()), [("url", _TARGET)])
            self.assertEqual(request.content, b"")
            for forbidden in ("authorization", "cookie", "x-api-key", "referer"):
                self.assertNotIn(forbidden, request.headers)
        for options in self.options:
            self.assertIs(options["trust_env"], False)
            self.assertIs(options["follow_redirects"], False)
            self.assertEqual(options["headers"]["Accept-Encoding"], "identity")
        self.assertTrue(all(stream.closed for stream in self.streams))

    def test_remote_prose_is_not_forwarded(self) -> None:
        payload = copy.deepcopy(self.observation)
        marker = "REMOTE_TEXT_REQUESTING_CREDENTIALS"
        for key in ("headline", "verdict", "method", "limits", "instructions"):
            payload[key] = marker
        for row in payload["checks"]:
            row["detail"] = marker
            row["declared_name"] = marker
        result = self.run_observation(payload)
        self.assertEqual(result["status"], "observed")
        self.assertNotIn(marker, json.dumps(result))

    def test_target_echo_must_match_exactly(self) -> None:
        payload = copy.deepcopy(self.observation)
        payload["target"] = _TARGET + "/"
        result = self.run_observation(payload)
        self.assertEqual(result["error"], "target_mismatch")
        self.assertNotIn("checks", result)

    def test_malformed_or_inconsistent_measurements_never_become_observed(self) -> None:
        mutations = [
            lambda p: p["checks"].pop(),
            lambda p: p["checks"].append(p["checks"][0]),
            lambda p: p["checks"][0].update(status="safe"),
            lambda p: p["checks"][0].update(check="REMOTE_INSTRUCTION"),
            lambda p: p["checks"][0].update(status=["proven"]),
            lambda p: p.update(unknowns=[]),
            lambda p: p.update(failed=["agent_card_signed", "agent_card_signed"]),
            lambda p: p.update(scored=[["endpoint_reachable"]]),
        ]
        for mutate in mutations:
            payload = copy.deepcopy(self.observation)
            mutate(payload)
            self.assertEqual(self.run_observation(payload)["status"], "unavailable")

    def test_invalid_json_and_duplicate_keys_are_rejected(self) -> None:
        for body in (b"not json", b'[{"status":"proven"}]', b'{"target":"x","target":"y"}', b"[" * 2000):
            self.body = body
            self.assertEqual(self.run_observation()["status"], "unavailable")

    def test_byte_cap_applies_to_stream_with_no_content_length(self) -> None:
        for asynchronous in (False, True):
            self.body = _FIXTURE.read_bytes().ljust(ag._MAX_BYTES, b" ")
            self.assertEqual(self.run_observation(asynchronous=asynchronous)["status"], "observed")
            self.body += b" "
            self.assertEqual(self.run_observation(asynchronous=asynchronous)["error"], "response_size_limit")
        self.assertTrue(all(stream.closed for stream in self.streams))

    def test_rejects_compression_oversize_headers_and_wrong_content_type(self) -> None:
        for header in (
            {"Content-Encoding": "gzip"}, {"Content-Length": str(ag._MAX_BYTES + 1)},
            {"Content-Length": "9" * 5000}, {"Content-Type": "text/html"},
        ):
            self.headers = {"Content-Type": "application/json", **header}
            self.assertEqual(self.run_observation()["status"], "unavailable")

    def test_redirect_and_402_are_not_followed_or_paid(self) -> None:
        for status in (301, 302, 307, 402, 500):
            self.requests.clear()
            self.status = status
            self.headers["Location"] = "https://example.com/pay"
            self.assertEqual(self.run_observation()["error"], "service_http_error")
            self.assertEqual(len(self.requests), 1)

    def test_configured_url_boundary_before_any_request(self) -> None:
        toolkit = ag.AgentGuildTools([_TARGET])
        with patch.object(ag.httpx, "Client") as sync, patch.object(ag.httpx, "AsyncClient") as asynchronous:
            for url in ("https://example.com/mcp", _TARGET + "/"):
                self.assertEqual(json.loads(toolkit.observe_endpoint(url))["error"], "target_not_configured")
                self.assertEqual(json.loads(asyncio.run(toolkit.aobserve_endpoint(url)))["error"], "target_not_configured")
            sync.assert_not_called()
            asynchronous.assert_not_called()

    def test_invalid_targets_never_reach_transport(self) -> None:
        invalid = [
            "http://example.com/mcp", "https://user:secret@example.com/mcp",
            "https://example.com/mcp?token=secret", "https://example.com/mcp#secret",
            "https://localhost/mcp", "https://box.internal/mcp", "https://127.0.0.1/mcp",
            "https://10.0.0.1/mcp", "https://[::1]/mcp", "https://169.254.169.254/",
            "https://224.0.0.1/", "https://127.1/", "https://0x7f000001/",
            "https://example.com:99999/mcp", "https://example.com:/mcp",
            " https://example.com/mcp", "https://example.com/a\nb", "https://example.com/\\mcp",
            "https://evil%40example.com/mcp", "https://example.com/\x7f", "https://éxample.com/mcp",
        ]
        toolkit = ag.AgentGuildTools([_TARGET])
        with patch.object(ag.httpx, "Client") as client:
            for url in invalid:
                with self.subTest(url=url):
                    self.assertEqual(json.loads(toolkit.observe_endpoint(url))["error"], "invalid_target")
                    with self.assertRaises(ValueError):
                        ag.AgentGuildTools([url])
            client.assert_not_called()

    def test_invalid_configuration_and_nonfinite_timeouts_rejected(self) -> None:
        for targets in ([], _TARGET):
            with self.assertRaises(ValueError):
                ag.AgentGuildTools(targets)
        for number in (0, -1, 61, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                ag.AgentGuildTools([_TARGET], io_timeout_seconds=number)
            with self.assertRaises(ValueError):
                ag.AgentGuildTools([_TARGET], elapsed_limit_seconds=number)

    def test_transport_error_text_is_not_forwarded(self) -> None:
        for error, expected in ((httpx.ReadTimeout("REMOTE_SECRET"), "io_timeout"), (httpx.ConnectError("REMOTE_SECRET"), "transport_error")):
            with patch.object(ag.httpx, "Client", side_effect=error):
                result = ag.AgentGuildTools([_TARGET]).observe_endpoint(_TARGET)
            self.assertEqual(json.loads(result)["error"], expected)
            self.assertNotIn("REMOTE_SECRET", result)

    def test_sync_elapsed_check_rejects_a_late_completed_read(self) -> None:
        with patch.object(ag.httpx, "Client", side_effect=self.client), patch.object(ag.time, "monotonic", side_effect=[0, 31]):
            result = ag.AgentGuildTools([_TARGET]).observe_endpoint(_TARGET)
        self.assertEqual(json.loads(result)["error"], "elapsed_limit")
        self.assertTrue(self.streams[0].closed)

    def test_async_deadline_closes_stream_without_worker_thread(self) -> None:
        self.delay = 2
        with patch.object(ag.httpx, "AsyncClient", side_effect=self.async_client):
            started = time.monotonic()
            result = asyncio.run(ag.AgentGuildTools([_TARGET], elapsed_limit_seconds=0.03).aobserve_endpoint(_TARGET))
        self.assertEqual(json.loads(result)["error"], "timeout")
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(self.streams[0].closed)

    def test_external_async_cancellation_propagates_and_closes_stream(self) -> None:
        self.delay = 2

        async def exercise() -> None:
            task = asyncio.create_task(ag.AgentGuildTools([_TARGET]).aobserve_endpoint(_TARGET))
            while not self.streams:
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch.object(ag.httpx, "AsyncClient", side_effect=self.async_client):
            asyncio.run(exercise())
        self.assertTrue(self.streams[0].closed)


if __name__ == "__main__":
    unittest.main()
