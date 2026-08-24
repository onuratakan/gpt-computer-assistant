---
name: guardrails-tool-authorization
description: Use when working with Upsonic's pre-tool-call authorization layer, GuardrailProvider implementations, AllowlistProvider, or custom OAP/APort-style guardrails. Trigger when a user asks to allow or deny tool calls before execution, audit whether an agent was authorized to call a tool, or integrate an external authorization service alongside SafetyEngine.
---

# `src/upsonic/guardrails/` — Pre-Tool-Call Authorization

Upsonic has two complementary control layers:

| Layer | Question | Upsonic surface |
| --- | --- | --- |
| Content safety | Is this input, output, or tool-call content safe? | `SafetyEngine`, `PolicyManager`, `ToolPolicyManager` |
| Tool authorization | Is this agent authorized to execute this tool call at all? | `GuardrailProvider` |

The guardrail layer runs immediately before a tool executes. It is intentionally
provider-agnostic: a project can use the built-in allowlist provider, an internal
authorization service, or an Open Agent Protocol / APort-style provider without
adding a hard dependency to Upsonic.

## Quick Start

```python
from upsonic import Agent, Task
from upsonic.guardrails import AllowlistProvider

agent = Agent(
    model="openai/gpt-4o",
    guardrail_provider=AllowlistProvider(allowed_tools=["search_docs"]),
)

task = Task("Search the docs", tools=[search_docs, write_file])
result = agent.do(task)
```

If the model calls `write_file`, Upsonic returns a tool result explaining that
the call was blocked. The tool itself is not executed and does not count against
the executed tool-call counter.

## Provider Interface

Custom providers receive a normalized `GuardrailRequest`:

```python
from upsonic.guardrails import GuardrailDecision, GuardrailRequest

class InternalPolicyProvider:
    name = "internal-policy"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name == "delete_customer":
            return GuardrailDecision.deny_decision(
                "policy.delete_customer",
                "Agents cannot delete customer records.",
                provider_name=self.name,
            )
        return GuardrailDecision.allow_decision(provider_name=self.name)
```

Async providers can implement `aevaluate(request)`.

`GuardrailRequest.tool_input` is an alias for `GuardrailRequest.arguments`, so
generic OAP-style providers that expect `.tool_name` and `.tool_input` can
consume Upsonic requests directly.

## Fail-Closed Behavior

Provider errors fail closed. If a provider raises, returns an invalid decision,
or does not expose `evaluate()` / `aevaluate()`, Upsonic blocks the tool call
and returns a structured denial message to the model.

## Common Upsonic Patterns

Guardrail providers are most useful when a tool has side effects, reaches a
sensitive system, or handles regulated data. The provider is optional; agents
without `guardrail_provider` keep the existing Upsonic execution path.

### Autonomous DevOps and Coding Agents

Upsonic autonomous agents can work inside a filesystem workspace and use shell
tools. A common setup is to allow read-only tools by default, then require a
guardrail decision for commands, edits, deploys, and destructive operations:

```python
from upsonic import AutonomousAgent
from upsonic.guardrails import AllowlistProvider

agent = AutonomousAgent(
    model="anthropic/claude-sonnet-4-5",
    workspace="./workspace",
    guardrail_provider=AllowlistProvider(
        allowed_tools=["read_file", "search_files", "grep_files", "edit_file"],
        denied_tools=["delete_file", "run_command"],
    ),
)
```

For hosted policy systems, map these tools to action-specific policies such as
file read/write, shell execution, repository pull requests, or release actions.

### Fintech and Merchant Operations

Upsonic is often used for merchant onboarding, document collection, risk
monitoring, settlement workflows, and communication. Guardrails let teams keep
low-risk analysis tools open while gating actions that move money, update
merchant status, send messages, or export customer data:

```python
merchant_ops_guardrail = AllowlistProvider(
    allowed_tools=[
        "classify_email",
        "extract_receipt",
        "analyze_merchant_document",
        "check_merchant_status",
    ],
    denied_tools=["send_payout", "refund_payment", "export_customer_data"],
)
```

### Document, Contract, and Research Agents

For OCR, contract analysis, RAG, and web research, a guardrail provider can
allow extraction and search while blocking exfiltration or outbound actions:

```python
contract_guardrail = AllowlistProvider(
    allowed_tools=["ocr_document", "extract_contract_terms", "search_knowledge_base"],
    denied_tools=["send_email", "share_document", "export_customer_data"],
)
```

### MCP and Provider-Native Tools

Guardrails run inside Upsonic immediately before a tool executes. Provider-native
builtin tools execute at the model provider, outside Upsonic's local tool
runner. When `guardrail_provider` is configured, Upsonic rejects those builtin
tools, including model-setting sources such as `openai_builtin_tools`, instead
of pretending it can authorize them. Use regular Upsonic tools or MCP wrappers
when you need deterministic pre-execution authorization.

External-execution tools are also rejected while `guardrail_provider` is active.
Those tools pause inside Upsonic and run out-of-band later, which means Upsonic
cannot guarantee a fresh authorization decision at the moment the external action
happens.

## External Providers

For regulated deployments, teams can plug in a provider backed by an external
authorization service. An OAP/APort-style provider can send the tool name,
arguments, agent identity, and task context to a policy endpoint, then map the
remote allow/deny decision into `GuardrailDecision`.

Upsonic does not ship that dependency by default. The contract stays small so
teams can keep tool authorization local, hosted, or vendor-neutral.

If you already use `aport-agent-guardrails`, its generic provider can be passed
directly after configuring `.aport/config.yaml` for hosted or local mode:

```python
from aport_guardrails.providers import OAPGuardrailProvider
from upsonic import Agent

agent = Agent(
    model="openai/gpt-4o",
    guardrail_provider=OAPGuardrailProvider(
        framework="upsonic",
        config_path=".aport/config.yaml",
    ),
)
```

Example with a small APort Verify provider after creating a hosted APort
passport and passport-bound setup key:

```python
import os
import requests

from upsonic import Agent
from upsonic.guardrails import GuardrailDecision, GuardrailReason, GuardrailRequest


class APortVerifyProvider:
    name = "aport"

    def __init__(self, agent_id, api_key=None, api_url="https://api.aport.io"):
        self.agent_id = agent_id
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")

    @classmethod
    def from_env(cls):
        return cls(
            agent_id=os.environ["APORT_AGENT_ID"],
            api_key=os.environ.get("APORT_API_KEY"),
            api_url=os.environ.get("APORT_API_URL", "https://api.aport.io"),
        )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.api_url}/api/verify/policy/mcp.tool.execute.v1",
            headers=headers,
            json={
                "context": {
                    "agent_id": self.agent_id,
                    "policy_id": "mcp.tool.execute.v1",
                    "server": "https://upsonic.ai/tools",
                    "tool": request.tool_name,
                    "parameters": dict(request.arguments),
                    "context": {"task": request.task_description},
                }
            },
            timeout=10,
        )
        response.raise_for_status()
        decision = response.json()["decision"]
        allow = decision.get("allow")
        if type(allow) is not bool:
            return GuardrailDecision.deny_decision(
                "oap.invalid_decision",
                "APort Verify response did not include a boolean allow value.",
                provider_name=self.name,
            )
        return GuardrailDecision(
            allow=allow,
            reasons=tuple(
                GuardrailReason(
                    code=reason.get("code", "oap.reason"),
                    message=reason.get("message", "APort decision"),
                    severity=reason.get("severity", "info"),
                )
                for reason in decision.get("reasons", ())
            ),
            provider_name=self.name,
        )


agent = Agent(
    model="openai/gpt-4o",
    guardrail_provider=APortVerifyProvider.from_env(),
)
```

This uses `mcp.tool.execute.v1` as a generic external-tool gate. For production,
map each Upsonic tool to the most specific APort policy pack for that action
category, such as file read/write, shell execution, web fetch, or repository
operations.
