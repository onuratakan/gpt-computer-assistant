"""Tests for MCP stdio server environment isolation (issue #621)."""

import os

import pytest

from upsonic.tools.mcp import MCPHandler


@pytest.fixture
def _leaky_env(monkeypatch):
    """Populate os.environ with secret-looking vars that must NOT leak."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-67890")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_faketoken000000")
    yield


class TestStdioEnvIsolation:
    """A stdio MCP server must not inherit the full ``os.environ``."""

    def test_secrets_not_leaked(self, _leaky_env):
        handler = MCPHandler(command="echo hello")
        env = handler.server_params.env

        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "GITHUB_TOKEN" not in env

    def test_safe_vars_preserved(self, _leaky_env):
        handler = MCPHandler(command="echo hello")
        env = handler.server_params.env

        # Essential runtime vars should still be present
        assert "PATH" in env
        assert "HOME" in env

    def test_user_env_merged(self, _leaky_env):
        handler = MCPHandler(
            command="echo hi", env={"MY_TOOL_KEY": "abc123"}
        )
        env = handler.server_params.env

        assert env.get("MY_TOOL_KEY") == "abc123"
        # User env merge must not re-introduce leaked secrets
        assert "OPENAI_API_KEY" not in env
