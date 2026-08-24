import copy
from types import SimpleNamespace

import time

import pytest

from upsonic.agent.agent import Agent
from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent
from upsonic.agent.deepagent.tools.subagent_toolkit import SubagentToolKit
from upsonic.agent.pipeline.manager import PipelineManager
from upsonic.guardrails import AllowlistProvider, GuardrailDecision
from upsonic.messages import ToolCallPart
from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.requirements import RunRequirement
from upsonic.run.tools.tools import ToolExecution
from upsonic.tools import ToolDefinition, ToolResult
from upsonic.tools.builtin_tools import WebSearchTool
from upsonic.tools.hitl import ConfirmationPause, PauseHandler, PausedToolCall
from upsonic.tools.orchestration import Orchestrator, PlanStep
from upsonic.tools.wrappers import AgentTool


def make_agent(provider):
    agent = Agent.__new__(Agent)
    agent.guardrail_provider = provider
    agent.name = "Test Agent"
    agent.agent_id_ = "agent-test"
    agent.metadata = {"env": "test"}
    agent.current_task = SimpleNamespace(description="test task")
    return agent


def make_tool_call(tool_name: str = "search") -> ToolCallPart:
    return ToolCallPart(
        tool_name=tool_name,
        args={"query": "hello"},
        tool_call_id="call-test",
    )


def make_tool_def() -> ToolDefinition:
    return ToolDefinition(
        name="search",
        description="Search the web",
        parameters_json_schema={"type": "object"},
    )


class FakeSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeOtel:
    def tool_span(self, *args, **kwargs):
        return FakeSpan()

    def set_tool_result(self, *args, **kwargs):
        return None


class FakeToolManager:
    def __init__(self, definitions=None, functions=None):
        self.definitions = definitions or [
            ToolDefinition(
                name="search",
                description="Search docs",
                parameters_json_schema={"type": "object"},
            ),
            ToolDefinition(
                name="write_file",
                description="Write a file",
                parameters_json_schema={"type": "object"},
            ),
        ]
        self.executed = []
        self.registry = SimpleNamespace(
            wrapped_tools={definition.name: object() for definition in self.definitions},
            registered_tools={
                name: SimpleNamespace(function=function)
                for name, function in (functions or {}).items()
            },
        )

    def get_tool_definitions(self):
        return self.definitions

    async def execute_tool(self, tool_name, args, metrics=None, tool_call_id=None):
        self.executed.append((tool_name, args))
        return ToolResult(
            tool_name=tool_name,
            content=f"executed {tool_name}",
            tool_call_id=tool_call_id,
        )


def make_executing_agent(provider=None):
    manager = FakeToolManager()
    agent = make_agent(provider)
    agent.run_id = None
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent._non_executed_tool_attempt_count = 0
    agent._tool_limit_reached = False
    agent._agent_run_output = None
    agent._tool_metrics = None
    agent._otel = FakeOtel()
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager
    return agent, manager


@pytest.mark.asyncio
async def test_authorize_tool_call_allows_when_provider_allows():
    agent = make_agent(AllowlistProvider(allowed_tools=["search"]))

    result = await agent._authorize_tool_call(make_tool_call("search"), make_tool_def())

    assert result is None


@pytest.mark.asyncio
async def test_authorize_tool_call_blocks_when_provider_denies():
    agent = make_agent(AllowlistProvider(allowed_tools=["search"]))

    result = await agent._authorize_tool_call(make_tool_call("write_file"), make_tool_def())

    assert result is not None
    assert result.tool_name == "write_file"
    assert "Tool blocked by guardrail provider" in result.content
    assert "not in the allowed tool list" in result.content


def test_authorization_model_dump_preserves_nested_validation_alias_path():
    from pydantic import AliasPath, BaseModel, Field

    class PolicyArgs(BaseModel):
        scope: str = Field(validation_alias=AliasPath("policy", "scope"))

    args = PolicyArgs.model_validate({"policy": {"scope": "all"}})

    assert Agent._guardrail_model_dump_for_authorization(args) == {
        "policy": {"scope": "all"}
    }


def test_authorization_arguments_are_isolated_from_provider_mutation():
    agent, manager = make_executing_agent(provider=AllowlistProvider(allowed_tools=["search"]))
    args = {"query": "hello", "options": {"scope": "all"}}

    authorization_args = agent._guardrail_authorization_arguments(manager, "search", args)
    authorization_args["options"]["scope"] = "mutated"

    assert args["options"]["scope"] == "all"


@pytest.mark.asyncio
async def test_authorize_tool_call_builds_context_for_provider():
    captured = {}

    class CapturingProvider:
        name = "capturing"

        def evaluate(self, request):
            captured["request"] = request
            return True

    agent = make_agent(CapturingProvider())

    result = await agent._authorize_tool_call(make_tool_call("search"), make_tool_def())

    assert result is None
    request = captured["request"]
    assert request.tool_name == "search"
    assert request.arguments == {"query": "hello"}
    assert request.tool_call_id == "call-test"
    assert request.tool_description == "Search the web"
    assert request.agent_name == "Test Agent"
    assert request.agent_id == "agent-test"
    assert request.task_description == "test task"
    assert request.metadata == {"env": "test"}


@pytest.mark.asyncio
async def test_execute_tool_calls_without_provider_preserves_normal_execution():
    agent, manager = make_executing_agent(provider=None)

    results = await agent._execute_tool_calls([make_tool_call("search")])

    assert len(results) == 1
    assert results[0].content == "executed search"
    assert manager.executed == [("search", {"query": "hello"})]
    assert agent._tool_call_count == 1


@pytest.mark.asyncio
async def test_parallel_guardrail_denials_do_not_count_as_executed():
    agent, manager = make_executing_agent(
        provider=AllowlistProvider(allowed_tools=["search"]),
    )

    results = await agent._execute_tool_calls(
        [make_tool_call("search"), make_tool_call("write_file")]
    )

    assert len(results) == 2
    assert manager.executed == [("search", {"query": "hello"})]
    assert agent._tool_call_count == 1
    assert any("Tool blocked by guardrail provider" in str(result.content) for result in results)


@pytest.mark.asyncio
async def test_guardrail_authorizes_default_expanded_arguments():
    captured = {}

    def export_records(scope="all"):
        return scope

    class ScopeProvider:
        name = "scope_provider"

        def evaluate(self, request):
            captured["arguments"] = request.arguments
            if request.arguments.get("scope") == "all":
                return GuardrailDecision.deny_decision("scope.denied", "scope=all is denied")
            return True

    definition = ToolDefinition(
        name="export_records",
        description="Export records",
        parameters_json_schema={"type": "object"},
    )
    manager = FakeToolManager(definitions=[definition], functions={"export_records": export_records})
    agent = make_executing_agent(provider=ScopeProvider())[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    results = await agent._execute_tool_calls([
        ToolCallPart(tool_name="export_records", args={}, tool_call_id="call-default")
    ])

    assert captured["arguments"] == {"scope": "all"}
    assert manager.executed == []
    assert "scope=all is denied" in str(results[0].content)


@pytest.mark.asyncio
async def test_guardrail_default_expansion_does_not_mutate_execution_kwargs():
    captured = {}

    def export_records(scope="all"):
        return scope

    class CapturingProvider:
        name = "capturing"

        def evaluate(self, request):
            captured["arguments"] = request.arguments
            return True

    definition = ToolDefinition(
        name="export_records",
        description="Export records",
        parameters_json_schema={"type": "object"},
    )
    manager = FakeToolManager(definitions=[definition], functions={"export_records": export_records})
    agent = make_executing_agent(provider=CapturingProvider())[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    await agent._execute_tool_calls([
        ToolCallPart(tool_name="export_records", args={}, tool_call_id="call-default")
    ])

    assert captured["arguments"] == {"scope": "all"}
    assert manager.executed == [("export_records", {})]


@pytest.mark.asyncio
async def test_guardrail_normalizes_optional_pydantic_arguments():
    from typing import Optional
    from pydantic import BaseModel

    captured = {}

    class ExportOptions(BaseModel):
        destructive: bool = True

    def export_records(options: Optional[ExportOptions] = None):
        return options

    class CapturingProvider:
        name = "capturing"

        def evaluate(self, request):
            captured["arguments"] = request.arguments
            return GuardrailDecision.deny_decision("test.stop", "stop")

    definition = ToolDefinition(
        name="export_records",
        description="Export records",
        parameters_json_schema={"type": "object"},
    )
    manager = FakeToolManager(definitions=[definition], functions={"export_records": export_records})
    agent = make_executing_agent(provider=CapturingProvider())[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    await agent._execute_tool_calls([
        ToolCallPart(tool_name="export_records", args={"options": {}}, tool_call_id="call-model")
    ])

    assert captured["arguments"] == {"options": {"destructive": True}}
    assert manager.executed == []


@pytest.mark.asyncio
async def test_guardrail_preserves_pydantic_schema_aliases_for_authorization():
    from pydantic import BaseModel, Field

    captured = {}

    class ExportOptions(BaseModel):
        scope: str = Field(validation_alias="accessScope", serialization_alias="serializedScope")

    def export_records(options: ExportOptions):
        return options

    class CapturingProvider:
        name = "capturing"

        def evaluate(self, request):
            captured["arguments"] = request.arguments
            return GuardrailDecision.deny_decision("test.stop", "stop")

    definition = ToolDefinition(
        name="export_records",
        description="Export records",
        parameters_json_schema={"type": "object"},
    )
    manager = FakeToolManager(definitions=[definition], functions={"export_records": export_records})
    agent = make_executing_agent(provider=CapturingProvider())[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    await agent._execute_tool_calls([
        ToolCallPart(
            tool_name="export_records",
            args={"options": {"accessScope": "all"}},
            tool_call_id="call-model",
        )
    ])

    assert captured["arguments"] == {"options": {"accessScope": "all"}}
    assert manager.executed == []


@pytest.mark.asyncio
async def test_guardrail_normalizes_list_pydantic_arguments():
    from pydantic import BaseModel

    captured = {}

    class ExportOptions(BaseModel):
        destructive: bool = True

    def export_records(options: list[ExportOptions]):
        return options

    class CapturingProvider:
        name = "capturing"

        def evaluate(self, request):
            captured["arguments"] = request.arguments
            return GuardrailDecision.deny_decision("test.stop", "stop")

    definition = ToolDefinition(
        name="export_records",
        description="Export records",
        parameters_json_schema={"type": "object"},
    )
    manager = FakeToolManager(definitions=[definition], functions={"export_records": export_records})
    agent = make_executing_agent(provider=CapturingProvider())[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    await agent._execute_tool_calls([
        ToolCallPart(tool_name="export_records", args={"options": [{}]}, tool_call_id="call-model")
    ])

    assert captured["arguments"] == {"options": [{"destructive": True}]}
    assert manager.executed == []


@pytest.mark.asyncio
async def test_guardrail_uses_resolved_manager_definition_for_duplicate_names():
    captured = {}

    class CapturingProvider:
        name = "capturing"

        def evaluate(self, request):
            captured["description"] = request.tool_description
            return True

    agent_definition = ToolDefinition(
        name="shared_tool",
        description="Agent-level implementation",
        parameters_json_schema={"type": "object"},
    )
    task_definition = ToolDefinition(
        name="shared_tool",
        description="Task-level implementation",
        parameters_json_schema={"type": "object"},
    )
    agent_manager = FakeToolManager(definitions=[agent_definition])
    task_manager = FakeToolManager(definitions=[task_definition])
    agent = make_agent(CapturingProvider())
    agent.run_id = None
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent._tool_limit_reached = False
    agent._agent_run_output = None
    agent._tool_metrics = None
    agent._otel = FakeOtel()
    agent.tool_manager = agent_manager
    agent.current_task = SimpleNamespace(
        description="test task",
        response_format=None,
        tool_manager=task_manager,
    )

    await agent._execute_tool_calls([
        ToolCallPart(tool_name="shared_tool", args={}, tool_call_id="call-duplicate")
    ])

    assert captured["description"] == "Agent-level implementation"
    assert agent_manager.executed == [("shared_tool", {})]
    assert task_manager.executed == []


@pytest.mark.asyncio
async def test_guardrail_runs_after_async_tool_policy_validation():
    class DynamicProvider:
        name = "dynamic"

        def __init__(self):
            self.allow = True

        def evaluate(self, request):
            if self.allow:
                return True
            return GuardrailDecision.deny_decision("policy.revoked", "policy revoked")

    class ValidationResult:
        disallowed_exception = None

        def should_block(self):
            return False

    class ToolPolicyPostManager:
        def __init__(self, provider):
            self.provider = provider

        def has_policies(self):
            return True

        async def execute_tool_call_validation_async(self, *, tool_call_info, check_type):
            self.provider.allow = False
            return ValidationResult()

    provider = DynamicProvider()
    agent, manager = make_executing_agent(provider=provider)
    agent.tool_policy_post_manager = ToolPolicyPostManager(provider)

    results = await agent._execute_tool_calls([make_tool_call("search")])

    assert manager.executed == []
    assert "policy revoked" in str(results[0].content)


@pytest.mark.asyncio
async def test_tool_execution_time_uses_tool_result_not_guardrail_latency():
    class TimedToolManager(FakeToolManager):
        async def execute_tool(self, tool_name, args, metrics=None, tool_call_id=None):
            self.executed.append((tool_name, args))
            return ToolResult(
                tool_name=tool_name,
                content=f"executed {tool_name}",
                tool_call_id=tool_call_id,
                execution_time=0.25,
            )

    agent = make_agent(AllowlistProvider(allowed_tools=["search"]))
    manager = TimedToolManager()
    output = AgentRunOutput()
    agent.run_id = None
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent._non_executed_tool_attempt_count = 0
    agent._tool_limit_reached = False
    agent._agent_run_output = output
    agent._tool_metrics = None
    agent._otel = FakeOtel()
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    await agent._execute_tool_calls([make_tool_call("search")])

    assert output.usage is not None
    assert output.usage.tool_execution_time == 0.25


@pytest.mark.asyncio
async def test_parallel_tool_execution_time_preserves_zero_duration_result():
    class CachedToolManager(FakeToolManager):
        async def execute_tool(self, tool_name, args, metrics=None, tool_call_id=None):
            self.executed.append((tool_name, args))
            return ToolResult(
                tool_name=tool_name,
                content=f"executed {tool_name}",
                tool_call_id=tool_call_id,
                execution_time=0.0,
            )

    agent = make_agent(AllowlistProvider(allowed_tools=["search"]))
    manager = CachedToolManager()
    output = AgentRunOutput()
    agent.run_id = None
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent._non_executed_tool_attempt_count = 0
    agent._tool_limit_reached = False
    agent._agent_run_output = output
    agent._tool_metrics = None
    agent._otel = FakeOtel()
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    await agent._execute_tool_calls([make_tool_call("search")])

    assert output.usage is not None
    assert output.usage.tool_execution_time == 0.0


def test_merged_execution_interval_duration_counts_staggered_parallel_wall_time():
    assert Agent._merged_execution_interval_duration([(0.0, 5.0), (4.0, 9.0)]) == 9.0


@pytest.mark.asyncio
async def test_hitl_resume_reauthorizes_final_arguments():
    called = []

    def write_file(**kwargs):
        called.append(kwargs)
        return "wrote"

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(AllowlistProvider(allowed_tools=["search"]))
    agent._resolve_tool_manager = lambda tool_name: manager

    result, executed, execution_time = await agent._execute_hitl_tool_directly(
        "write_file",
        {"path": "secret.txt", "content": "deny me"},
        "call-hitl",
    )

    assert "Tool blocked by guardrail provider" in result
    assert executed is False
    assert execution_time is None
    assert called == []


@pytest.mark.asyncio
async def test_hitl_resume_denial_is_injected_without_execution_count():
    called = []

    def write_file(**kwargs):
        called.append(kwargs)
        return "wrote"

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(AllowlistProvider(allowed_tools=["search"]))
    agent._resolve_tool_manager = lambda tool_name: manager
    agent._tool_call_count = 0

    tool_execution = ToolExecution(
        tool_call_id="call-hitl",
        tool_name="write_file",
        tool_args={"path": "secret.txt", "content": "deny me"},
        requires_confirmation=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    requirement.confirm()
    output = AgentRunOutput(tools=[])

    await agent._inject_hitl_results(output, [requirement])

    assert called == []
    assert agent._tool_call_count == 0
    assert output.tool_call_count == 0
    assert output.tools == []
    assert tool_execution.result_injected is True
    assert "Tool blocked by guardrail provider" in str(output.chat_history[-1].parts[0].content)


@pytest.mark.asyncio
async def test_hitl_resume_execution_failure_is_non_executed_attempt():
    def write_file(**kwargs):
        raise RuntimeError("disk unavailable")

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(AllowlistProvider(allowed_tools=["write_file"]))
    output = AgentRunOutput(tools=[])
    agent._agent_run_output = output
    agent._resolve_tool_manager = lambda tool_name: manager

    result, executed, execution_time = await agent._execute_hitl_tool_directly(
        "write_file",
        {"path": "out.txt"},
        "call-hitl",
    )

    assert "disk unavailable" in result
    assert executed is False
    assert execution_time is not None
    assert agent._non_executed_tool_attempt_count == 1
    assert output.non_executed_tool_attempt_count == 1


@pytest.mark.asyncio
async def test_hitl_resume_tool_execution_time_excludes_reauthorization_latency(monkeypatch):
    clock = [0.0]

    class SlowPolicyProvider:
        name = "slow_policy"

        def evaluate(self, request):
            clock[0] += 10.0
            return True

    def write_file(**kwargs):
        clock[0] += 0.25
        return "wrote"

    monkeypatch.setattr(time, "time", lambda: clock[0])

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(SlowPolicyProvider())
    agent._resolve_tool_manager = lambda tool_name: manager

    result, executed, execution_time = await agent._execute_hitl_tool_directly(
        "write_file",
        {"path": "out.txt"},
        "call-hitl",
    )

    assert result == "wrote"
    assert executed is True
    assert execution_time == 0.25


@pytest.mark.asyncio
async def test_hitl_resume_execution_failure_records_tool_runtime_without_execution_count():
    def write_file(**kwargs):
        raise RuntimeError("disk unavailable")

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(AllowlistProvider(allowed_tools=["write_file"]))
    agent._resolve_tool_manager = lambda tool_name: manager
    agent._tool_call_count = 0

    tool_execution = ToolExecution(
        tool_call_id="call-hitl",
        tool_name="write_file",
        tool_args={"path": "out.txt"},
        requires_confirmation=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    requirement.confirm()
    output = AgentRunOutput(tools=[])

    await agent._inject_hitl_results(output, [requirement])

    assert agent._tool_call_count == 0
    assert output.tool_call_count == 0
    assert output.tools == []
    assert output.usage is not None
    assert output.usage.tool_execution_time is not None
    assert output.usage.tool_execution_time >= 0


@pytest.mark.asyncio
async def test_guarded_hitl_resume_without_provider_fails_closed():
    called = []

    def write_file(**kwargs):
        called.append(kwargs)
        return "wrote"

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(provider=None)
    agent._resolve_tool_manager = lambda tool_name: manager
    agent._tool_call_count = 0

    tool_execution = ToolExecution(
        tool_call_id="call-hitl",
        tool_name="write_file",
        tool_args={"path": "secret.txt", "content": "deny me"},
        requires_confirmation=True,
        requires_guardrail_authorization=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    requirement.confirm()
    output = AgentRunOutput(tools=[])

    await agent._inject_hitl_results(output, [requirement])

    assert called == []
    assert agent._tool_call_count == 0
    assert agent._guardrail_denied_tool_call_count == 1
    assert output.guardrail_denied_tool_call_count == 1
    assert output.tools == []
    assert tool_execution.result_injected is True
    assert "guarded HITL continuation requires a guardrail provider" in str(
        output.chat_history[-1].parts[0].content
    )


@pytest.mark.asyncio
async def test_falsey_provider_still_marks_paused_calls_guarded():
    class FalseyProvider:
        name = "falsey"

        def __bool__(self):
            return False

    manager = PipelineManager.__new__(PipelineManager)
    manager.agent = SimpleNamespace(guardrail_provider=FalseyProvider())
    manager.task = None
    manager.debug = False

    async def save_session(output):
        return None

    manager._save_session = save_session
    output = AgentRunOutput()

    await manager._handle_confirmation_pause(
        output,
        ConfirmationPause(paused_calls=[
            PausedToolCall(
                tool_name="write_file",
                tool_args={"path": "secret.txt"},
                tool_call_id="call-falsey",
                requires_confirmation=True,
            )
        ]),
    )

    requirement = output.requirements[0]
    assert requirement.tool_execution.requires_guardrail_authorization is True


@pytest.mark.asyncio
async def test_hitl_resume_respects_exhausted_tool_call_limit():
    called = []

    def write_file(**kwargs):
        called.append(kwargs)
        return "wrote"

    manager = FakeToolManager()
    manager.registry.registered_tools["write_file"] = SimpleNamespace(function=write_file)
    agent = make_agent(provider=None)
    agent._resolve_tool_manager = lambda tool_name: manager
    agent.tool_call_limit = 1
    agent._tool_call_count = 1
    agent._guardrail_denied_tool_call_count = 0
    agent._non_executed_tool_attempt_count = 0

    tool_execution = ToolExecution(
        tool_call_id="call-hitl",
        tool_name="write_file",
        tool_args={"path": "secret.txt", "content": "deny me"},
        requires_confirmation=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    requirement.confirm()
    output = AgentRunOutput(tools=[], tool_call_count=1)

    await agent._inject_hitl_results(output, [requirement])

    assert called == []
    assert output.tools == []
    assert output.tool_limit_reached is True
    assert tool_execution.result_injected is True
    assert "Tool call limit of 1 reached" in str(output.chat_history[-1].parts[0].content)


@pytest.mark.asyncio
async def test_external_execution_result_rejected_under_guardrail_provider():
    agent = make_agent(AllowlistProvider(allowed_tools=["external_transfer"]))
    agent._tool_call_count = 0

    tool_execution = ToolExecution(
        tool_call_id="call-external",
        tool_name="external_transfer",
        tool_args={"amount": 100},
        result="sent",
        external_execution_required=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    output = AgentRunOutput(tools=[])

    await agent._inject_hitl_results(output, [requirement])

    assert agent._tool_call_count == 0
    assert agent._guardrail_denied_tool_call_count == 1
    assert output.guardrail_denied_tool_call_count == 1
    assert output.tools == []
    assert tool_execution.result_injected is True
    assert "external_execution results cannot be accepted" in str(
        output.chat_history[-1].parts[0].content
    )


def test_build_model_request_parameters_rejects_builtin_tools_with_guardrail():
    agent = Agent.__new__(Agent)
    agent.guardrail_provider = AllowlistProvider(allowed_tools=["search"])
    agent._tool_limit_reached = False
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent.tool_manager = SimpleNamespace(get_tool_definitions=lambda: [])
    agent.agent_builtin_tools = [WebSearchTool()]

    task = SimpleNamespace(
        tool_manager=None,
        task_builtin_tools=[],
        response_format=None,
    )

    with pytest.raises(ValueError, match="provider-native builtin tools"):
        agent._build_model_request_parameters(task)


def test_build_model_request_parameters_rejects_settings_builtin_tools_with_guardrail():
    agent = Agent.__new__(Agent)
    agent.guardrail_provider = AllowlistProvider(allowed_tools=["search"])
    agent._tool_limit_reached = False
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent._guardrail_denied_tool_call_count = 0
    agent.tool_manager = SimpleNamespace(get_tool_definitions=lambda: [])
    agent.agent_builtin_tools = []
    agent.model = SimpleNamespace(
        settings={
            "openai_builtin_tools": [{"type": "web_search"}],
        }
    )

    task = SimpleNamespace(
        tool_manager=None,
        task_builtin_tools=[],
        response_format=None,
    )

    with pytest.raises(ValueError, match="openai_builtin_tools"):
        agent._build_model_request_parameters(task)


def test_build_model_request_parameters_rejects_groq_implicit_web_search_with_guardrail():
    agent = Agent.__new__(Agent)
    agent.guardrail_provider = AllowlistProvider(allowed_tools=["search"])
    agent._tool_limit_reached = False
    agent.tool_call_limit = 100
    agent._tool_call_count = 0
    agent._guardrail_denied_tool_call_count = 0
    agent._non_executed_tool_attempt_count = 0
    agent.tool_manager = SimpleNamespace(get_tool_definitions=lambda: [])
    agent.agent_builtin_tools = []
    agent.model = SimpleNamespace(
        settings={},
        profile=SimpleNamespace(groq_always_has_web_search_builtin_tool=True),
    )

    task = SimpleNamespace(
        tool_manager=None,
        task_builtin_tools=[],
        response_format=None,
    )

    with pytest.raises(ValueError, match="groq_always_has_web_search_builtin_tool"):
        agent._build_model_request_parameters(task)


@pytest.mark.asyncio
async def test_guardrail_rejects_external_execution_tools():
    definition = ToolDefinition(
        name="external_transfer",
        description="External transfer",
        parameters_json_schema={"type": "object"},
    )
    manager = FakeToolManager(definitions=[definition], functions={"external_transfer": lambda: "sent"})
    manager.registry.registered_tools["external_transfer"].config = SimpleNamespace(external_execution=True)
    agent = make_executing_agent(provider=AllowlistProvider(allowed_tools=["external_transfer"]))[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager

    results = await agent._execute_tool_calls([
        ToolCallPart(tool_name="external_transfer", args={}, tool_call_id="call-external")
    ])

    assert manager.executed == []
    assert "external_execution tools cannot be authorized" in str(results[0].content)
    assert agent._tool_call_count == 0
    assert agent._guardrail_denied_tool_call_count == 1


@pytest.mark.asyncio
async def test_guardrail_denials_consume_attempt_limit_without_execution_count():
    agent, manager = make_executing_agent(
        provider=AllowlistProvider(allowed_tools=["write_file"]),
    )
    agent.tool_call_limit = 1

    first = await agent._execute_tool_calls([make_tool_call("search")])
    second = await agent._execute_tool_calls([make_tool_call("search")])

    assert manager.executed == []
    assert "Tool blocked by guardrail provider" in str(first[0].content)
    assert "Tool call limit of 1 reached" in str(second[0].content)
    assert agent._tool_call_count == 0
    assert agent._guardrail_denied_tool_call_count == 1
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_sequential_denial_stops_remaining_calls_when_attempt_limit_reached():
    manager = FakeToolManager(definitions=[
        ToolDefinition(
            name="write_file",
            description="Write a file",
            parameters_json_schema={"type": "object"},
            sequential=True,
        ),
        ToolDefinition(
            name="search",
            description="Search docs",
            parameters_json_schema={"type": "object"},
            sequential=True,
        ),
    ])
    agent = make_executing_agent(provider=AllowlistProvider(allowed_tools=["search"]))[0]
    agent.tool_manager = manager
    agent.current_task = SimpleNamespace(description="test task", response_format=None, tool_manager=None)
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager
    agent.tool_call_limit = 1

    results = await agent._execute_tool_calls([
        make_tool_call("write_file"),
        make_tool_call("search"),
    ])

    assert manager.executed == []
    assert "Tool blocked by guardrail provider" in str(results[0].content)
    assert "Tool call limit of 1 reached" in str(results[1].content)
    assert agent._tool_call_count == 0
    assert agent._guardrail_denied_tool_call_count == 1
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_parallel_policy_blocks_consume_attempt_limit_without_execution_count():
    class BlockingValidationResult:
        disallowed_exception = None

        def should_block(self):
            return True

        def get_final_message(self):
            return "blocked by policy"

    class BlockingToolPolicyPostManager:
        def has_policies(self):
            return True

        async def execute_tool_call_validation_async(self, *, tool_call_info, check_type):
            return BlockingValidationResult()

    agent, manager = make_executing_agent(provider=None)
    agent.tool_call_limit = 1
    agent.tool_policy_post_manager = BlockingToolPolicyPostManager()

    first = await agent._execute_tool_calls([make_tool_call("search")])
    second = await agent._execute_tool_calls([make_tool_call("search")])

    assert manager.executed == []
    assert "blocked by policy" in str(first[0].content)
    assert "Tool call limit of 1 reached" in str(second[0].content)
    assert agent._tool_call_count == 0
    assert agent._non_executed_tool_attempt_count == 1
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_parallel_unknown_tools_consume_attempt_limit_without_execution_count():
    agent, manager = make_executing_agent(provider=None)
    agent.tool_call_limit = 1

    def raise_unknown_tool(tool_name):
        raise ValueError(f"Unknown tool: {tool_name}")

    agent._resolve_tool_manager = raise_unknown_tool

    first = await agent._execute_tool_calls([make_tool_call("missing_tool")])
    second = await agent._execute_tool_calls([make_tool_call("missing_tool")])

    assert manager.executed == []
    assert "Unknown tool" in str(first[0].content)
    assert "Tool call limit of 1 reached" in str(second[0].content)
    assert agent._tool_call_count == 0
    assert agent._non_executed_tool_attempt_count == 1
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_parallel_batch_does_not_execute_past_remaining_attempt_limit():
    agent, manager = make_executing_agent(provider=None)
    agent.tool_call_limit = 1

    results = await agent._execute_tool_calls([
        ToolCallPart(
            tool_name="search",
            args={"query": "first"},
            tool_call_id="call-first",
        ),
        ToolCallPart(
            tool_name="write_file",
            args={"query": "second"},
            tool_call_id="call-second",
        ),
    ])

    assert manager.executed == [("search", {"query": "first"})]
    assert results[0].content == "executed search"
    assert "Tool call limit of 1 reached" in str(results[1].content)
    assert agent._tool_call_count == 1
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_parallel_output_tool_does_not_consume_attempt_limit_slot():
    from pydantic import BaseModel
    from upsonic.output import DEFAULT_OUTPUT_TOOL_NAME

    class StructuredResult(BaseModel):
        answer: str

    agent, manager = make_executing_agent(provider=None)
    agent.tool_call_limit = 1
    agent.current_task = SimpleNamespace(
        description="test task",
        response_format=StructuredResult,
        tool_manager=None,
    )

    results = await agent._execute_tool_calls([
        ToolCallPart(
            tool_name=DEFAULT_OUTPUT_TOOL_NAME,
            args={"answer": "done"},
            tool_call_id="call-output",
        ),
        ToolCallPart(
            tool_name="search",
            args={"query": "first"},
            tool_call_id="call-first",
        ),
        ToolCallPart(
            tool_name="write_file",
            args={"query": "second"},
            tool_call_id="call-second",
        ),
    ])

    assert manager.executed == [("search", {"query": "first"})]
    assert results[0].tool_name == DEFAULT_OUTPUT_TOOL_NAME
    assert results[1].content == "executed search"
    assert "Tool call limit of 1 reached" in str(results[2].content)
    assert agent._tool_call_count == 1
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_parallel_pause_commits_completed_attempt_accounting():
    class PausingToolManager(FakeToolManager):
        async def execute_tool(self, tool_name, args, metrics=None, tool_call_id=None):
            if tool_name == "write_file":
                raise ConfirmationPause(paused_calls=[
                    PausedToolCall(
                        tool_name=tool_name,
                        tool_args=args,
                        tool_call_id=tool_call_id,
                        requires_confirmation=True,
                    )
                ])
            return await super().execute_tool(tool_name, args, metrics=metrics, tool_call_id=tool_call_id)

    manager = PausingToolManager()
    agent = make_executing_agent(provider=None)[0]
    agent.tool_manager = manager
    agent._get_combined_tool_definitions = manager.get_tool_definitions
    agent._resolve_tool_manager = lambda tool_name: manager
    output = AgentRunOutput(tools=[])
    output.response = SimpleNamespace(parts=[])
    agent._agent_run_output = output

    with pytest.raises(ConfirmationPause):
        await agent._execute_tool_calls([
            ToolCallPart(
                tool_name="search",
                args={"query": "first"},
                tool_call_id="call-first",
            ),
            ToolCallPart(
                tool_name="write_file",
                args={"query": "second"},
                tool_call_id="call-second",
            ),
        ])

    assert manager.executed == [("search", {"query": "first"})]
    assert agent._tool_call_count == 1
    assert output.tool_call_count == 1
    assert len(output.tools or []) == 1
    assert output.tools[0].tool_name == "search"
    assert output.chat_history[0] == output.response
    assert output.chat_history[1].parts[0].tool_name == "search"


def test_reset_tool_execution_counters_clears_guardrail_denials():
    agent = Agent.__new__(Agent)
    agent._tool_call_count = 2
    agent._guardrail_denied_tool_call_count = 1
    agent._non_executed_tool_attempt_count = 1
    agent._tool_limit_reached = True

    agent._reset_tool_execution_counters()

    assert agent._tool_call_count == 0
    assert agent._guardrail_denied_tool_call_count == 0
    assert agent._non_executed_tool_attempt_count == 0
    assert agent._tool_limit_reached is False


def test_record_guardrail_denial_preserves_persisted_count():
    agent = Agent.__new__(Agent)
    agent._guardrail_denied_tool_call_count = 2
    output = AgentRunOutput(guardrail_denied_tool_call_count=2)

    agent._record_guardrail_denial(output)

    assert agent._guardrail_denied_tool_call_count == 3
    assert output.guardrail_denied_tool_call_count == 3


def test_agent_run_output_serializes_guardrail_denied_attempt_count():
    output = AgentRunOutput(guardrail_denied_tool_call_count=3)

    restored = AgentRunOutput.from_dict(output.to_dict())

    assert restored.guardrail_denied_tool_call_count == 3


def test_agent_run_output_serializes_non_executed_attempt_count():
    output = AgentRunOutput(non_executed_tool_attempt_count=2)

    restored = AgentRunOutput.from_dict(output.to_dict())

    assert restored.non_executed_tool_attempt_count == 2


def test_tool_execution_serializes_guardrail_authorization_requirement():
    tool_execution = ToolExecution(
        tool_name="write_file",
        tool_args={"path": "secret.txt"},
        requires_confirmation=True,
        result_injected=True,
        requires_guardrail_authorization=True,
    )

    restored = ToolExecution.from_dict(tool_execution.to_dict())

    assert restored.requires_guardrail_authorization is True
    assert restored.result_injected is True


def test_agent_level_agent_tool_inherits_guardrail_provider():
    provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None)
    captured = {}
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = provider
    parent.tools = [child]
    parent.canvas = None
    parent.enable_thinking_tool = False
    parent.tool_manager = SimpleNamespace(
        register_tools=lambda tools, task=None, agent_instance=None, **kwargs: (
            captured.setdefault("tools", tools),
            {},
        )[1]
    )
    parent._validate_tools_with_policy_pre = lambda *args, **kwargs: None

    parent._register_agent_tools()

    assert child.guardrail_provider is None
    assert len(captured["tools"]) == 1
    assert isinstance(captured["tools"][0], AgentTool)
    assert captured["tools"][0].agent is child
    assert captured["tools"][0].guardrail_provider is provider


def test_dynamic_agent_tool_inherits_guardrail_provider():
    provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None)
    captured = {}
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = provider
    parent.tools = []
    parent.enable_thinking_tool = False
    parent.agent_builtin_tools = []
    parent.registered_agent_tools = {}
    parent.tool_manager = SimpleNamespace(
        register_tools=lambda tools, task=None, agent_instance=None, **kwargs: (
            captured.setdefault("tools", tools),
            {},
        )[1]
    )
    parent._validate_tools_with_policy_pre = lambda *args, **kwargs: None

    parent.add_tools(child)

    assert child.guardrail_provider is None
    assert len(captured["tools"]) == 1
    assert isinstance(captured["tools"][0], AgentTool)
    assert captured["tools"][0].agent is child
    assert captured["tools"][0].guardrail_provider is provider


def test_task_level_agent_tool_inherits_guardrail_provider():
    provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None)
    captured = {}
    task_manager = SimpleNamespace(
        register_tools=lambda tools, task=None, agent_instance=None, **kwargs: (
            captured.setdefault("tools", tools),
            {},
        )[1]
    )
    task = SimpleNamespace(
        tools=[child],
        enable_thinking_tool=False,
        enable_reasoning_tool=False,
        skills=None,
        registered_task_tools={},
        _ensure_tool_manager=lambda: task_manager,
    )
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = provider
    parent.enable_thinking_tool = False
    parent.enable_reasoning_tool = False
    parent.tool_call_limit = 100
    parent._tool_call_count = 0
    parent.registered_agent_tools = {}
    parent._validate_tools_with_policy_pre = lambda *args, **kwargs: None

    parent._setup_task_tools(task)

    assert child.guardrail_provider is None
    assert len(captured["tools"]) == 1
    assert isinstance(captured["tools"][0], AgentTool)
    assert captured["tools"][0].agent is child
    assert captured["tools"][0].guardrail_provider is provider


def test_agent_tool_inheritance_binds_parent_policy_to_wrapper_only():
    provider = AllowlistProvider(allowed_tools=["child"])
    grandchild = SimpleNamespace(name="grandchild", do_async=lambda: None, guardrail_provider=None, tools=[])
    child = SimpleNamespace(
        name="child",
        do_async=lambda: None,
        guardrail_provider=None,
        tools=[grandchild],
    )
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = provider

    inherited_tools = parent._inherit_guardrail_provider_for_agent_tools([child])

    assert child.guardrail_provider is None
    assert grandchild.guardrail_provider is None
    assert len(inherited_tools) == 1
    assert isinstance(inherited_tools[0], AgentTool)
    assert inherited_tools[0].agent is child
    assert inherited_tools[0].guardrail_provider is provider


def test_inherited_agent_tool_provider_is_scoped_to_wrapper_when_child_is_reused():
    provider_a = AllowlistProvider(allowed_tools=["read_file"])
    provider_b = AllowlistProvider(allowed_tools=["write_file"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None, tools=[])

    wrapper_a = Agent._agent_tool_with_inherited_guardrail_provider(child, provider_a)
    wrapper_b = Agent._agent_tool_with_inherited_guardrail_provider(child, provider_b)

    assert child.guardrail_provider is None
    assert isinstance(wrapper_a, AgentTool)
    assert isinstance(wrapper_b, AgentTool)
    assert wrapper_a.agent is child
    assert wrapper_b.agent is child
    assert wrapper_a.guardrail_provider is provider_a
    assert wrapper_b.guardrail_provider is provider_b


@pytest.mark.asyncio
async def test_inherited_agent_tool_resolves_current_parent_policy_at_execution():
    provider_a = AllowlistProvider(allowed_tools=["read_file"])
    provider_b = AllowlistProvider(allowed_tools=["write_file"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        async def do_async(self, task, return_output=True):
            calls.append(self.guardrail_provider)
            return SimpleNamespace(output="ok")

    parent = Agent.__new__(Agent)
    parent.guardrail_provider = provider_a
    wrapper = parent._inherit_guardrail_provider_for_agent_tools([ChildAgent()])[0]

    parent.guardrail_provider = provider_b

    assert wrapper.guardrail_provider is provider_b
    await wrapper.execute("delegate this")
    assert calls == [provider_b]


@pytest.mark.asyncio
async def test_agent_tool_registered_before_parent_policy_still_resolves_policy_at_execution():
    provider = AllowlistProvider(allowed_tools=["write_file"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        async def do_async(self, task, return_output=True):
            calls.append(self.guardrail_provider)
            return SimpleNamespace(output="ok")

    parent = Agent.__new__(Agent)
    parent.guardrail_provider = None
    wrapper = parent._inherit_guardrail_provider_for_agent_tools([ChildAgent()])[0]

    parent.guardrail_provider = provider

    assert isinstance(wrapper, AgentTool)
    assert wrapper.guardrail_provider is provider
    await wrapper.execute("delegate this")
    assert calls == [provider]


def test_sync_only_agent_tool_inherits_parent_policy():
    provider = AllowlistProvider(allowed_tools=["child"])

    class SyncChildAgent:
        name = "child"
        guardrail_provider = None

        def do(self, task):
            return "ok"

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(
        SyncChildAgent(),
        provider,
    )

    assert isinstance(inherited_tool, AgentTool)
    assert inherited_tool.guardrail_provider is provider


def test_agent_id_metadata_without_execution_method_is_not_agent_tool():
    provider = AllowlistProvider(allowed_tools=["child"])

    class PlainToolObject:
        name = "plain"
        agent_id_ = "metadata-only"
        guardrail_provider = None

        def public_method(self):
            return "ok"

    plain_tool = PlainToolObject()

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(
        plain_tool,
        provider,
    )

    assert inherited_tool is plain_tool


def test_callable_provider_object_is_not_treated_as_dynamic_getter():
    class CallableProvider(AllowlistProvider):
        def __call__(self):
            raise AssertionError("provider object must not be invoked during inheritance")

    provider = CallableProvider(allowed_tools=["child"])

    class SyncChildAgent:
        name = "child"
        guardrail_provider = None

        def do(self, task):
            return "ok"

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(
        SyncChildAgent(),
        provider,
    )

    assert isinstance(inherited_tool, AgentTool)
    assert inherited_tool.guardrail_provider is provider


def test_dynamic_agent_tool_provider_reference_does_not_pickle_parent_agent():
    import cloudpickle
    import threading

    provider = AllowlistProvider(allowed_tools=["child"])

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        def do(self, task):
            return "ok"

    parent = Agent.__new__(Agent)
    parent.guardrail_provider = provider
    parent.unpickleable_client = threading.Lock()

    wrapper = parent._inherit_guardrail_provider_for_agent_tools([ChildAgent()])[0]
    restored = cloudpickle.loads(cloudpickle.dumps(wrapper))

    assert isinstance(restored, AgentTool)
    assert restored.guardrail_provider.allowed_tools == {"child"}


def test_deserialized_dynamic_agent_tool_rebinds_to_current_parent_policy():
    import cloudpickle
    import threading
    from upsonic.tools import ToolManager

    original_provider = AllowlistProvider(allowed_tools=["original"])
    current_provider = AllowlistProvider(allowed_tools=["current"])

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        def do(self, task):
            return "ok"

    original_parent = Agent.__new__(Agent)
    original_parent.guardrail_provider = original_provider
    original_parent.unpickleable_client = threading.Lock()

    restored = cloudpickle.loads(
        cloudpickle.dumps(
            original_parent._inherit_guardrail_provider_for_agent_tools([ChildAgent()])[0]
        )
    )

    current_parent = Agent.__new__(Agent)
    current_parent.guardrail_provider = current_provider
    current_parent.tool_manager = ToolManager()

    task = SimpleNamespace(tool_manager=ToolManager())
    task.tool_manager.registry.registered_tools[restored.name] = restored

    current_parent._rebind_guardrail_provider_references(task)

    assert restored.guardrail_provider is current_provider


def test_legacy_agent_tool_provider_state_is_migrated_on_read():
    provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None)
    tool = AgentTool(child)

    del tool.__dict__["_guardrail_provider"]
    del tool.__dict__["_guardrail_provider_getter"]
    tool.__dict__["guardrail_provider"] = provider

    assert tool.guardrail_provider is provider
    assert tool.__dict__["_guardrail_provider"] is provider
    assert tool.__dict__["_guardrail_provider_getter"] is None
    assert "guardrail_provider" not in tool.__dict__


def test_existing_agent_tool_inherits_parent_policy_on_new_wrapper():
    provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None)
    original_tool = AgentTool(child)

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(original_tool, provider)

    assert original_tool.guardrail_provider is None
    assert isinstance(inherited_tool, AgentTool)
    assert inherited_tool is not original_tool
    assert inherited_tool.agent is child
    assert inherited_tool.guardrail_provider is provider


def test_existing_agent_tool_dynamic_inheritance_preserves_wrapper_type():
    provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=None)

    class CustomAgentTool(AgentTool):
        pass

    original_tool = CustomAgentTool(child)
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = None

    inherited_tool = parent._inherit_guardrail_provider_for_agent_tools([original_tool])[0]
    parent.guardrail_provider = provider

    assert inherited_tool is not original_tool
    assert isinstance(inherited_tool, CustomAgentTool)
    assert inherited_tool.agent is child
    assert inherited_tool.guardrail_provider is provider
    assert original_tool.guardrail_provider is None


def test_existing_agent_tool_with_child_provider_is_not_overwritten_by_parent():
    parent_provider = AllowlistProvider(allowed_tools=["parent"])
    child_provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=child_provider)
    original_tool = AgentTool(child)

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(original_tool, parent_provider)

    assert inherited_tool is original_tool
    assert inherited_tool.guardrail_provider is None
    assert inherited_tool._agent_for_execution() is child
    assert child.guardrail_provider is child_provider


@pytest.mark.asyncio
async def test_inherited_agent_tool_uses_child_provider_added_after_registration():
    parent_provider = AllowlistProvider(allowed_tools=["parent"])
    child_provider = AllowlistProvider(allowed_tools=["child"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        async def do_async(self, task, return_output=True):
            calls.append(self.guardrail_provider)
            return SimpleNamespace(output="ok")

    child = ChildAgent()
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = parent_provider
    wrapper = parent._inherit_guardrail_provider_for_agent_tools([AgentTool(child)])[0]

    child.guardrail_provider = child_provider

    await wrapper.execute("delegate this")

    assert calls == [child_provider]
    assert wrapper.guardrail_provider is parent_provider


@pytest.mark.asyncio
async def test_explicit_agent_tool_provider_still_overrides_child_provider():
    wrapper_provider = AllowlistProvider(allowed_tools=["wrapper"])
    child_provider = AllowlistProvider(allowed_tools=["child"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = child_provider

        async def do_async(self, task, return_output=True):
            calls.append(self.guardrail_provider)
            return SimpleNamespace(output="ok")

    await AgentTool(ChildAgent(), guardrail_provider=wrapper_provider).execute("delegate this")

    assert calls == [wrapper_provider]


@pytest.mark.asyncio
async def test_explicit_agent_tool_provider_setter_clears_inherited_status():
    parent_provider = AllowlistProvider(allowed_tools=["parent"])
    wrapper_provider = AllowlistProvider(allowed_tools=["wrapper"])
    child_provider = AllowlistProvider(allowed_tools=["child"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        async def do_async(self, task, return_output=True):
            calls.append(self.guardrail_provider)
            return SimpleNamespace(output="ok")

    child = ChildAgent()
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = parent_provider
    wrapper = parent._inherit_guardrail_provider_for_agent_tools([AgentTool(child)])[0]

    wrapper.guardrail_provider = wrapper_provider
    child.guardrail_provider = child_provider

    await wrapper.execute("delegate this")

    assert calls == [wrapper_provider]


@pytest.mark.asyncio
async def test_legacy_dynamic_agent_tool_recovers_inherited_status():
    parent_provider = AllowlistProvider(allowed_tools=["parent"])
    child_provider = AllowlistProvider(allowed_tools=["child"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        async def do_async(self, task, return_output=True):
            calls.append(self.guardrail_provider)
            return SimpleNamespace(output="ok")

    child = ChildAgent()
    parent = Agent.__new__(Agent)
    parent.guardrail_provider = parent_provider
    wrapper = parent._inherit_guardrail_provider_for_agent_tools([AgentTool(child)])[0]
    del wrapper.__dict__["_guardrail_provider_inherited"]

    child.guardrail_provider = child_provider

    await wrapper.execute("delegate this")

    assert calls == [child_provider]
    assert wrapper.__dict__["_guardrail_provider_inherited"] is True


def test_explicit_agent_tool_provider_is_not_overwritten_by_parent():
    parent_provider = AllowlistProvider(allowed_tools=["parent"])
    explicit_provider = AllowlistProvider(allowed_tools=["child"])
    child = SimpleNamespace(name="child", do_async=lambda: None, guardrail_provider=explicit_provider, tools=[])

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(child, parent_provider)

    assert inherited_tool is child
    assert child.guardrail_provider is explicit_provider


def test_falsey_child_provider_is_not_overwritten_by_parent():
    class FalseyProvider(AllowlistProvider):
        def __bool__(self):
            return False

    parent_provider = AllowlistProvider(allowed_tools=["parent"])
    child_provider = FalseyProvider(allowed_tools=["child"])
    grandchild = SimpleNamespace(name="grandchild", do_async=lambda: None, guardrail_provider=None, tools=[])
    child = SimpleNamespace(
        name="child",
        do_async=lambda: None,
        guardrail_provider=child_provider,
        tools=[grandchild],
    )

    inherited_tool = Agent._agent_tool_with_inherited_guardrail_provider(child, parent_provider)

    assert inherited_tool is child
    assert child.guardrail_provider is child_provider
    assert grandchild.guardrail_provider is None


@pytest.mark.asyncio
async def test_inherited_agent_tool_executes_with_parent_policy_without_mutating_child():
    provider = AllowlistProvider(allowed_tools=["search"])
    calls = []

    class ChildAgent:
        name = "child"
        guardrail_provider = None

        async def do_async(self, task, return_output=True):
            calls.append((self.guardrail_provider, task.description))
            return SimpleNamespace(output="ok")

    child = ChildAgent()
    tool = AgentTool(child, guardrail_provider=provider)

    result = await tool.execute("delegate this")

    assert result == "ok"
    assert calls == [(provider, "delegate this")]
    assert child.guardrail_provider is None


@pytest.mark.asyncio
async def test_deepagent_subagent_task_uses_parent_policy_without_mutating_child():
    provider = AllowlistProvider(allowed_tools=["search"])
    calls = []

    class ChildAgent:
        name = "general-purpose"
        role = None
        goal = None
        system_prompt = None
        guardrail_provider = None

        async def do_async(self, task):
            calls.append((self.guardrail_provider, task.description))
            return "ok"

    child = ChildAgent()
    toolkit = SubagentToolKit(
        parent_agent=SimpleNamespace(
            subagents=[child],
            guardrail_provider=provider,
        )
    )

    result = await toolkit.task("delegate this")

    assert result == "ok"
    assert calls == [(provider, "delegate this")]
    assert child.guardrail_provider is None


def test_pause_handler_preserves_inner_paused_calls():
    pause = ConfirmationPause(
        paused_calls=[
            PausedToolCall(
                tool_name="write_file",
                tool_args={"path": "secret.txt"},
                tool_call_id="inner-call",
                requires_confirmation=True,
            )
        ]
    )
    PauseHandler().attach_paused_call(
        pause,
        tool_name="plan_and_execute",
        args={"thought": {"plan": []}},
        tool_call_id="outer-call",
        tool_obj=None,
    )

    assert len(pause.paused_calls) == 1
    assert pause.paused_calls[0].tool_name == "write_file"
    assert pause.paused_calls[0].tool_call_id == "inner-call"


@pytest.mark.asyncio
async def test_orchestrator_routes_steps_through_guarded_execution_path():
    guarded_calls = []

    async def guarded_execute(tool_calls):
        guarded_calls.extend(tool_calls)
        return [SimpleNamespace(content="blocked by guardrail")]

    def write_file(**kwargs):
        return "wrote"

    agent = SimpleNamespace(
        guardrail_provider=AllowlistProvider(allowed_tools=["plan_and_execute"]),
        enable_reasoning_tool=False,
        _execute_tool_calls=guarded_execute,
    )
    orchestrator = Orchestrator(
        agent_instance=agent,
        task=SimpleNamespace(description="test task"),
        wrapped_tools={"write_file": write_file},
    )

    result = await orchestrator._execute_single_step(
        PlanStep(
            tool_name="write_file",
            parameters={"path": "secret.txt"},
        )
    )

    assert result == "blocked by guardrail"
    assert len(guarded_calls) == 1
    assert guarded_calls[0].tool_name == "write_file"
    assert guarded_calls[0].args_as_dict() == {"path": "secret.txt"}


@pytest.mark.asyncio
async def test_orchestrator_reserves_outer_attempt_before_guarded_steps():
    class LimitedAgent:
        enable_reasoning_tool = False
        tool_call_limit = 1
        _tool_call_count = 0
        _guardrail_denied_tool_call_count = 0
        _non_executed_tool_attempt_count = 0
        _tool_limit_reached = False
        _agent_run_output = None

        def _tool_attempt_count(self):
            return (
                self._tool_call_count
                + self._guardrail_denied_tool_call_count
                + self._non_executed_tool_attempt_count
            )

        async def _execute_tool_calls(self, tool_calls):
            raise AssertionError("nested tool should not execute after reserving outer attempt")

    def write_file(**kwargs):
        return "wrote"

    agent = LimitedAgent()
    agent.guardrail_provider = AllowlistProvider(allowed_tools=["plan_and_execute", "write_file"])
    orchestrator = Orchestrator(
        agent_instance=agent,
        task=SimpleNamespace(description="test task"),
        wrapped_tools={"write_file": write_file},
    )

    result = await orchestrator._execute_single_step(
        PlanStep(
            tool_name="write_file",
            parameters={"path": "secret.txt"},
        )
    )

    assert result == "Tool call limit of 1 reached. Cannot execute more tools."
    assert agent._tool_limit_reached is True


@pytest.mark.asyncio
async def test_orchestrator_uses_live_parent_for_guarded_task_steps():
    provider_a = AllowlistProvider(allowed_tools=["plan_and_execute", "write_file"])
    provider_b = AllowlistProvider(allowed_tools=["plan_and_execute"])
    guarded_calls = []

    class ParentAgent:
        enable_reasoning_tool = False
        tool_call_limit = None
        _agent_run_output = None

        def __init__(self):
            self._guardrail_provider = provider_a

        @property
        def guardrail_provider(self):
            return self._guardrail_provider

        @guardrail_provider.setter
        def guardrail_provider(self, provider):
            self._guardrail_provider = provider

        async def _execute_tool_calls(self, tool_calls):
            guarded_calls.append((self, self.guardrail_provider, tool_calls[0].tool_name))
            return [SimpleNamespace(content="denied by current parent")]

    parent = ParentAgent()
    setup_time_copy = copy.copy(parent)
    parent.guardrail_provider = provider_b

    orchestrator = Orchestrator(
        agent_instance=setup_time_copy,
        task=SimpleNamespace(description="test task"),
        wrapped_tools={"write_file": lambda **kwargs: "wrote"},
        live_agent_instance=parent,
    )

    result = await orchestrator._execute_single_step(
        PlanStep(
            tool_name="write_file",
            parameters={"path": "secret.txt"},
        )
    )

    assert result == "denied by current parent"
    assert guarded_calls == [(parent, provider_b, "write_file")]


def test_autonomous_agent_accepts_guardrail_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = AllowlistProvider(allowed_tools=["read_file"])

    agent = AutonomousAgent(
        model="openai/gpt-4o",
        enable_filesystem=False,
        enable_shell=False,
        guardrail_provider=provider,
    )

    assert agent.guardrail_provider is provider
