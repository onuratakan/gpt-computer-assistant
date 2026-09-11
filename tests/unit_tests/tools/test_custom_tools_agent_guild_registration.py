"""Run in an actual Upsonic checkout with native dependencies installed."""

import asyncio
import inspect
import json
import unittest
from unittest.mock import patch

from upsonic.tools import ToolManager
from upsonic.tools.custom_tools import AgentGuildTools
from upsonic.tools.custom_tools import agent_guild as ag


class NativeRegistrationTests(unittest.TestCase):
    def test_modes_register_only_the_intended_tool_and_required_url(self) -> None:
        for asynchronous, expected in ((False, "observe_endpoint"), (True, "aobserve_endpoint")):
            with self.subTest(use_async=asynchronous):
                toolkit = AgentGuildTools(["https://example.com/mcp"], use_async=asynchronous)
                manager = ToolManager()
                registered = manager.register_tools([toolkit])
                self.assertEqual(set(registered), {expected})
                self.assertEqual([f.__name__ for f in toolkit.functions], [expected])
                self.assertEqual(inspect.iscoroutinefunction(toolkit.functions[0]), asynchronous)
                schema = registered[expected].schema.json_schema
                self.assertEqual(set(schema["properties"]), {"url"})
                self.assertEqual(schema["required"], ["url"])
                self.assertEqual(manager.collect_instructions(), [])
                self.assertEqual(manager.register_tools([toolkit]), {})

    def test_two_instances_keep_operator_selected_targets_independent(self) -> None:
        one = AgentGuildTools(["https://example.com/one"])
        two = AgentGuildTools(["https://example.com/two"])
        self.assertNotEqual(one._allowed_targets, two._allowed_targets)
        self.assertIn("target_not_configured", one.observe_endpoint("https://example.com/two"))
        self.assertIn("target_not_configured", two.observe_endpoint("https://example.com/one"))

    def test_native_wrapper_preserves_structured_unavailable_result(self) -> None:
        for asynchronous, name in ((False, "observe_endpoint"), (True, "aobserve_endpoint")):
            manager = ToolManager()
            manager.register_tools([AgentGuildTools(["https://example.com/selected"], use_async=asynchronous)])
            result = asyncio.run(manager.execute_tool(name, {"url": "https://example.com/unselected"}))
            self.assertTrue(result.success)
            self.assertEqual(json.loads(result.content["func"])["status"], "unavailable")
            self.assertEqual(json.loads(result.content["func"])["error"], "target_not_configured")

    def test_framework_timeout_tracks_helper_configuration(self) -> None:
        for elapsed, io_timeout in ((60.0, 10.0), (0.03, 0.01), (30.0, 10.0)):
            with self.subTest(elapsed=elapsed, io_timeout=io_timeout):
                toolkit = AgentGuildTools(
                    ["https://example.com/selected"],
                    elapsed_limit_seconds=elapsed, io_timeout_seconds=io_timeout,
                )
                registered = ToolManager().register_tools([toolkit])
                config = registered["observe_endpoint"].config
                self.assertEqual(config.timeout, elapsed + io_timeout + 1.0)
                self.assertEqual(config.max_retries, 0)
                self.assertFalse(config.cache_results)

    def test_async_helper_timeout_survives_native_wrapper_as_structured_result(self) -> None:
        cancelled = []

        async def waiting_fetch(target: str, io_timeout: float, elapsed_limit: float) -> str:
            try:
                await asyncio.sleep(2)
                return "unexpected completion"
            finally:
                cancelled.append(True)

        toolkit = AgentGuildTools(
            ["https://example.com/selected"], use_async=True,
            elapsed_limit_seconds=0.03, io_timeout_seconds=0.01,
        )
        manager = ToolManager()
        manager.register_tools([toolkit])
        with patch.object(ag, "_afetch", side_effect=waiting_fetch) as fetch:
            result = asyncio.run(manager.execute_tool("aobserve_endpoint", {"url": "https://example.com/selected"}))
        fetch.assert_called_once_with("https://example.com/selected", 0.01, 0.03)
        self.assertEqual(cancelled, [True])
        self.assertTrue(result.success)
        observation = json.loads(result.content["func"])
        self.assertEqual(observation["status"], "unavailable")
        self.assertEqual(observation["error"], "timeout")
        self.assertNotIn("checks", observation)
