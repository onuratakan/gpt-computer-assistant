"""Public Task integration with real Upsonic classes and mocked HTTP transport.

No model or external service call is needed.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from upsonic.tools import ToolManager
from upsonic.tools.custom_tools import AgentGuildTools
from upsonic.tools.custom_tools import agent_guild as ag


_REAL_CLIENT = httpx.Client
_REAL_ASYNC_CLIENT = httpx.AsyncClient
_TARGET = "https://example.com/mcp"


class _ResponseStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body
        self.closed = False

    def __iter__(self):
        yield self.body

    async def __aiter__(self):
        yield self.body

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


class PublicTaskIntegrationTests(unittest.TestCase):
    def test_public_task_tools_register_and_execute_with_fake_http(self):
        # Exercise the documented public lazy import and constructor, including
        # native model rebuilding/validation; do not patch Task or its imports.
        from upsonic import Task

        checks = [
            {"check": "endpoint_reachable", "status": "proven"},
            {"check": "protocol_handshake", "status": "proven"},
            {"check": "agent_card_resolves", "status": "proven"},
            {"check": "agent_card_signed", "status": "failed"},
            {"check": "payment_claim_holds", "status": "unknown"},
            {"check": "independent_evidence", "status": "unknown"},
        ]
        body = json.dumps({
            "target": _TARGET,
            "checks": checks,
            "failed": ["agent_card_signed"],
            "unknowns": ["payment_claim_holds", "independent_evidence"],
            "scored": [row["check"] for row in checks[:4]],
        }).encode()

        for use_async, name in ((False, "observe_endpoint"), (True, "aobserve_endpoint")):
            with self.subTest(use_async=use_async):
                requests = []
                streams = []

                def handler(request):
                    requests.append(request)
                    stream = _ResponseStream(body)
                    streams.append(stream)
                    return httpx.Response(
                        200, headers={"Content-Type": "application/json"}, stream=stream,
                    )

                def client(**kwargs):
                    return _REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

                def async_client(**kwargs):
                    return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

                toolkit = AgentGuildTools(allowed_targets=[_TARGET], use_async=use_async)
                task = Task(
                    description="Inspect the configured endpoint and preserve unknown checks.",
                    tools=[toolkit],
                )
                self.assertIs(task.tools[0], toolkit)
                self.assertIsNone(task.agent)

                manager = ToolManager()
                registered = manager.register_tools(task.tools, task=task)
                self.assertEqual(set(registered), {name})

                with patch.object(ag.httpx, "Client", side_effect=client), patch.object(
                    ag.httpx, "AsyncClient", side_effect=async_client,
                ):
                    result = asyncio.run(manager.execute_tool(
                        name, {"url": _TARGET}, tool_call_id="public-task-observation",
                    ))
                    self.assertTrue(result.success, result.error)
                    self.assertEqual(result.tool_call_id, "public-task-observation")
                    observation = json.loads(result.content["func"])
                    self.assertEqual(observation["status"], "observed")
                    self.assertEqual(observation["target"], _TARGET)
                    self.assertEqual(observation["checks"], checks)
                    self.assertEqual(
                        observation["unknowns"], ["payment_claim_holds", "independent_evidence"],
                    )

                    # Passing the toolkit through Task must retain its operator
                    # configuration, including rejection before HTTP for another URL.
                    refused = asyncio.run(manager.execute_tool(
                        name, {"url": "https://example.com/unconfigured"},
                    ))
                    self.assertTrue(refused.success, refused.error)
                    self.assertEqual(
                        json.loads(refused.content["func"])["error"], "target_not_configured",
                    )

                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].method, "GET")
                self.assertEqual(str(requests[0].url.copy_with(query=None)), ag._SERVICE)
                self.assertEqual(list(requests[0].url.params.multi_items()), [("url", _TARGET)])
                self.assertEqual(requests[0].content, b"")
                self.assertTrue(all(stream.closed for stream in streams))
