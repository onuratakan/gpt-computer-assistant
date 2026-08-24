"""Minimal APort Verify provider for Upsonic.

This example intentionally keeps APort optional. Create a hosted APort passport
and a passport-bound setup key, then run with:

    export APORT_AGENT_ID="ap_..."
    export APORT_API_KEY="apk_..."

The provider uses APort's generic MCP tool execution policy as a simple
external-tool authorization gate. Production deployments can map different
Upsonic tools to more specific APort policy packs when needed.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from upsonic.guardrails import GuardrailDecision, GuardrailReason, GuardrailRequest


class APortVerifyProvider:
    """GuardrailProvider that delegates tool authorization to APort Verify."""

    name = "aport"

    def __init__(
        self,
        *,
        agent_id: str,
        api_key: str | None = None,
        api_url: str = "https://api.aport.io",
        pack_id: str = "mcp.tool.execute.v1",
        server: str = "https://upsonic.ai/tools",
        timeout: float = 10.0,
    ) -> None:
        self.agent_id = agent_id
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.pack_id = pack_id
        self.server = server
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "APortVerifyProvider":
        agent_id = os.environ.get("APORT_AGENT_ID")
        if not agent_id:
            raise ValueError("APORT_AGENT_ID is required for APortVerifyProvider")

        return cls(
            agent_id=agent_id,
            api_key=os.environ.get("APORT_API_KEY"),
            api_url=os.environ.get("APORT_API_URL", "https://api.aport.io"),
        )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.api_url}/api/verify/policy/{self.pack_id}",
            headers=headers,
            json={"context": self._context(request)},
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        decision = payload.get("decision") if isinstance(payload, dict) else None
        if not isinstance(decision, dict):
            decision = payload if isinstance(payload, dict) else {}

        reasons = tuple(self._reason(reason) for reason in decision.get("reasons", ()))
        allow = decision.get("allow")
        if type(allow) is not bool:
            return GuardrailDecision.deny_decision(
                "oap.invalid_decision",
                "APort Verify response did not include a boolean allow value.",
                provider_name=self.name,
            )
        return GuardrailDecision(
            allow=allow,
            reasons=reasons,
            provider_name=self.name,
        )

    def _context(self, request: GuardrailRequest) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "policy_id": self.pack_id,
            "server": self.server,
            "tool": request.tool_name,
            "parameters": dict(request.arguments),
            "context": {
                "agent_name": request.agent_name,
                "task_description": request.task_description,
                "tool_call_id": request.tool_call_id,
            },
        }

    @staticmethod
    def _reason(reason: Any) -> GuardrailReason:
        if not isinstance(reason, dict):
            return GuardrailReason(
                code="oap.reason",
                message=str(reason),
                severity="info",
            )
        return GuardrailReason(
            code=str(reason.get("code", "oap.reason")),
            message=str(reason.get("message", reason.get("code", "APort decision"))),
            severity=str(reason.get("severity", "info")),
        )


def build_guardrailed_agent():
    from upsonic import Agent

    return Agent(
        model="openai/gpt-4o",
        guardrail_provider=APortVerifyProvider.from_env(),
    )
