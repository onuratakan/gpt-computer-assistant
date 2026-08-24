"""
Upsonic Tools System

A comprehensive, modular tool handling system for AI agents that supports:
- Function tools with decorators
- Class-based tools and toolkits
- Agent-as-tool functionality
- MCP (Model Context Protocol) tools
- Deferred and external tool execution
- Tool orchestration and planning
- Rich behavioral configuration (caching, confirmation, hooks, etc.)
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.tools.base import (
        Tool,
        ToolKit,
        ToolDefinition,
        ToolResult,
        ToolMetadata,
        ToolValidationError,
        DocstringFormat,
        ObjectJsonSchema,
    )
    from upsonic.tools.config import (
        tool,
        ToolConfig,
        ToolHooks,
    )
    from upsonic.tools.metrics import (
        ToolMetrics,
    )
    from upsonic.tools.schema import (
        FunctionSchema,
        function_schema,
        SchemaGenerationError,
    )
    from upsonic.tools.normalizer import (
        ToolNormalizer,
        NormalizationResult,
    )
    from upsonic.tools.registry import ToolRegistry
    from upsonic.tools.execution import ToolWrapper
    from upsonic.tools.hitl import (
        PausedToolCall,
        PauseHandler,
        ConfirmationPause,
        UserInputPause,
        ExternalExecutionPause,
    )
    from upsonic.tools.wrappers import (
        FunctionTool,
        AgentTool,
    )
    from upsonic.tools.orchestration import (
        PlanStep,
        AnalysisResult,
        Thought,
        ExecutionResult,
        plan_and_execute,
        Orchestrator,
        OrchestratorLifecycle,
    )
    from upsonic.tools.mcp import (
        MCPTool,
        MCPHandler,
    )
    from upsonic.tools.builtin_tools import (
        AbstractBuiltinTool,
        WebSearchTool,
        WebSearchUserLocation,
        CodeExecutionTool,
        UrlContextTool,
        WebSearch,
        WebRead,
    )

def _get_base_classes() -> Dict[str, Any]:
    """Lazy import of base classes."""
    from upsonic.tools.base import (
        Tool,
        ToolKit,
        ToolDefinition,
        ToolResult,
        ToolMetadata,
        ToolValidationError,
        DocstringFormat,
        ObjectJsonSchema,
    )

    return {
        'Tool': Tool,
        'ToolKit': ToolKit,
        'ToolDefinition': ToolDefinition,
        'ToolResult': ToolResult,
        'ToolMetadata': ToolMetadata,
        'ToolValidationError': ToolValidationError,
        'DocstringFormat': DocstringFormat,
        'ObjectJsonSchema': ObjectJsonSchema,
    }

def _get_config_classes() -> Dict[str, Any]:
    """Lazy import of config classes."""
    from upsonic.tools.config import (
        tool,
        ToolConfig,
        ToolHooks,
    )

    return {
        'tool': tool,
        'ToolConfig': ToolConfig,
        'ToolHooks': ToolHooks,
    }

def _get_metrics_classes() -> Dict[str, Any]:
    """Lazy import of metrics classes."""
    from upsonic.tools.metrics import (
        ToolMetrics,
    )

    return {
        'ToolMetrics': ToolMetrics,
    }

def _get_schema_classes() -> Dict[str, Any]:
    """Lazy import of schema classes."""
    from upsonic.tools.schema import (
        FunctionSchema,
        function_schema,
        SchemaGenerationError,
    )

    return {
        'FunctionSchema': FunctionSchema,
        'function_schema': function_schema,
        'SchemaGenerationError': SchemaGenerationError,
    }

def _get_normalizer_classes() -> Dict[str, Any]:
    """Lazy import of normalizer classes."""
    from upsonic.tools.normalizer import (
        ToolNormalizer,
        NormalizationResult,
    )

    return {
        'ToolNormalizer': ToolNormalizer,
        'NormalizationResult': NormalizationResult,
    }

def _get_registry_classes() -> Dict[str, Any]:
    """Lazy import of registry classes."""
    from upsonic.tools.registry import ToolRegistry

    return {
        'ToolRegistry': ToolRegistry,
    }

def _get_execution_classes() -> Dict[str, Any]:
    """Lazy import of execution classes."""
    from upsonic.tools.execution import ToolWrapper

    return {
        'ToolWrapper': ToolWrapper,
    }

def _get_hitl_classes() -> Dict[str, Any]:
    """Lazy import of HITL classes."""
    from upsonic.tools.hitl import (
        PausedToolCall,
        PauseHandler,
        ConfirmationPause,
        UserInputPause,
        ExternalExecutionPause,
    )

    return {
        'PausedToolCall': PausedToolCall,
        'PauseHandler': PauseHandler,
        'ConfirmationPause': ConfirmationPause,
        'UserInputPause': UserInputPause,
        'ExternalExecutionPause': ExternalExecutionPause,
    }

def _get_wrapper_classes() -> Dict[str, Any]:
    """Lazy import of wrapper classes."""
    from upsonic.tools.wrappers import (
        FunctionTool,
        AgentTool,
    )

    return {
        'FunctionTool': FunctionTool,
        'AgentTool': AgentTool,
    }

def _get_orchestration_classes() -> Dict[str, Any]:
    """Lazy import of orchestration classes."""
    from upsonic.tools.orchestration import (
        PlanStep,
        AnalysisResult,
        Thought,
        ExecutionResult,
        plan_and_execute,
        Orchestrator,
        OrchestratorLifecycle,
    )

    return {
        'PlanStep': PlanStep,
        'AnalysisResult': AnalysisResult,
        'Thought': Thought,
        'ExecutionResult': ExecutionResult,
        'plan_and_execute': plan_and_execute,
        'Orchestrator': Orchestrator,
        'OrchestratorLifecycle': OrchestratorLifecycle,
    }

def _get_mcp_classes() -> Dict[str, Any]:
    """Lazy import of MCP classes."""
    from upsonic.tools.mcp import (
        MCPTool,
        MCPHandler,
        MultiMCPHandler,
        SSEClientParams,
        StreamableHTTPClientParams,
        prepare_command,
    )

    return {
        'MCPTool': MCPTool,
        'MCPHandler': MCPHandler,
        'MultiMCPHandler': MultiMCPHandler,
        'SSEClientParams': SSEClientParams,
        'StreamableHTTPClientParams': StreamableHTTPClientParams,
        'prepare_command': prepare_command,
    }

def _get_builtin_classes() -> Dict[str, Any]:
    """Lazy import of builtin classes."""
    from upsonic.tools.builtin_tools import (
        AbstractBuiltinTool,
        WebSearchTool,
        WebSearchUserLocation,
        CodeExecutionTool,
        UrlContextTool,
        WebSearch,
        WebRead,
    )

    return {
        'AbstractBuiltinTool': AbstractBuiltinTool,
        'WebSearchTool': WebSearchTool,
        'WebSearchUserLocation': WebSearchUserLocation,
        'CodeExecutionTool': CodeExecutionTool,
        'UrlContextTool': UrlContextTool,
        'WebSearch': WebSearch,
        'WebRead': WebRead,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]

    config_classes = _get_config_classes()
    if name in config_classes:
        return config_classes[name]

    metrics_classes = _get_metrics_classes()
    if name in metrics_classes:
        return metrics_classes[name]

    schema_classes = _get_schema_classes()
    if name in schema_classes:
        return schema_classes[name]

    normalizer_classes = _get_normalizer_classes()
    if name in normalizer_classes:
        return normalizer_classes[name]

    registry_classes = _get_registry_classes()
    if name in registry_classes:
        return registry_classes[name]

    execution_classes = _get_execution_classes()
    if name in execution_classes:
        return execution_classes[name]

    hitl_classes = _get_hitl_classes()
    if name in hitl_classes:
        return hitl_classes[name]

    wrapper_classes = _get_wrapper_classes()
    if name in wrapper_classes:
        return wrapper_classes[name]

    orchestration_classes = _get_orchestration_classes()
    if name in orchestration_classes:
        return orchestration_classes[name]

    mcp_classes = _get_mcp_classes()
    if name in mcp_classes:
        return mcp_classes[name]

    builtin_classes = _get_builtin_classes()
    if name in builtin_classes:
        return builtin_classes[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class ToolManager:
    """High-level facade composing five collaborators."""

    def __init__(self) -> None:
        from upsonic.tools.normalizer import ToolNormalizer
        from upsonic.tools.registry import ToolRegistry
        from upsonic.tools.execution import ToolWrapper
        from upsonic.tools.hitl import PauseHandler
        from upsonic.tools.orchestration import OrchestratorLifecycle

        self.normalizer = ToolNormalizer()
        self.registry = ToolRegistry()
        self.wrapper = ToolWrapper(registry=self.registry)
        self.pause_handler = PauseHandler()
        self.orchestrator_lifecycle = OrchestratorLifecycle(registry=self.registry)

    def register_tools(
        self,
        tools: list,
        task: Optional['Task'] = None,
        agent_instance: Optional[Any] = None,
        live_agent_instance: Optional[Any] = None,
    ) -> Dict[str, 'Tool']:
        """Register tools, wrap them, and update the orchestrator state."""
        if not tools:
            return {}

        result = self.normalizer.normalize(tools, self.registry.raw_object_ids)
        new_tools = self.registry.add(result)

        for name, tool_obj in new_tools.items():
            self.registry.store_wrapped(name, self.wrapper.wrap(tool_obj))

        self.orchestrator_lifecycle.maybe_create(
            new_tools,
            task,
            agent_instance,
            live_agent_instance,
        )
        if task is not None:
            self.orchestrator_lifecycle.update_context(task)

        return new_tools

    def remove_tools(
        self,
        tools: Union[Any, List[Any]],
        registered_tools: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[Any]]:
        """Remove tools and refresh orchestrator state."""
        removed, originals = self.registry.remove(tools)
        self.orchestrator_lifecycle.maybe_discard(removed)
        return removed, originals

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        metrics: Optional['ToolMetrics'] = None,
        tool_call_id: Optional[str] = None,
    ) -> 'ToolResult':
        """Execute a tool by name using the pre-wrapped executor."""
        from upsonic.tools.base import ToolResult
        from upsonic.tools.hitl import (
            ConfirmationPause,
            ExternalExecutionPause,
            UserInputPause,
        )

        wrapped = self.registry.get_wrapped(tool_name)
        if not wrapped:
            raise ValueError(f"Tool '{tool_name}' not found or not wrapped")

        if not tool_call_id:
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

        validation_error: Optional[str] = self._validate_required_args(tool_name, args)
        if validation_error is not None:
            return ToolResult(
                tool_name=tool_name,
                content=validation_error,
                tool_call_id=tool_call_id,
                success=False,
                error=validation_error,
            )

        start_time = time.time()
        try:
            if tool_name == 'plan_and_execute':
                from upsonic.tools.orchestration import Thought
                if 'thought' in args:
                    thought_data = args['thought']
                    if isinstance(thought_data, dict):
                        thought = Thought(**thought_data)
                    else:
                        thought = thought_data
                else:
                    thought = Thought(**args)
                result = await wrapped(thought)
            else:
                result = await wrapped(**args)

            execution_time = time.time() - start_time
            return ToolResult(
                tool_name=tool_name,
                content=result,
                tool_call_id=tool_call_id,
                success=True,
                execution_time=execution_time,
            )

        except (ConfirmationPause, UserInputPause, ExternalExecutionPause) as e:
            self.pause_handler.attach_paused_call(
                e,
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                tool_obj=self.registry.get(tool_name),
            )
            raise

        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                content=str(e),
                tool_call_id=tool_call_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def get_tool_definitions(self) -> List['ToolDefinition']:
        """Get definitions for all registered tools."""
        return self.registry.all_definitions()

    def collect_instructions(self) -> List[str]:
        """Collect all active instructions from toolkits and individual tools."""
        return self.registry.collect_instructions()

    def _validate_required_args(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Validate that all required arguments are present before executing a tool."""
        tool = self.registry.get(tool_name)
        if not tool or not tool.schema:
            return None

        json_schema: Dict[str, Any] = tool.schema.json_schema
        required_params: list[str] = json_schema.get("required", [])

        missing: list[str] = [p for p in required_params if p not in args]
        if missing:
            return (
                f"Missing required argument(s) for '{tool_name}': {', '.join(missing)}. "
                f"This usually means the model's response was truncated (max_tokens too low). "
                f"Please retry with all required parameters: {required_params}"
            )
        return None


__all__ = [
    'Tool',
    'ToolKit',
    'ToolDefinition',
    'ToolResult',
    'ToolMetadata',
    'ToolValidationError',
    'DocstringFormat',
    'ObjectJsonSchema',

    'tool',
    'ToolConfig',
    'ToolHooks',

    'ToolMetrics',

    'FunctionSchema',
    'function_schema',
    'SchemaGenerationError',

    'ToolNormalizer',
    'NormalizationResult',
    'ToolRegistry',
    'ToolWrapper',
    'PauseHandler',
    'OrchestratorLifecycle',

    'PausedToolCall',
    'ConfirmationPause',
    'UserInputPause',
    'ExternalExecutionPause',

    'FunctionTool',
    'AgentTool',

    'PlanStep',
    'AnalysisResult',
    'Thought',
    'ExecutionResult',
    'plan_and_execute',
    'Orchestrator',

    'MCPTool',
    'MCPHandler',
    'MultiMCPHandler',
    'SSEClientParams',
    'StreamableHTTPClientParams',
    'prepare_command',

    'ToolManager',

    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'UrlContextTool',
    'WebSearch',
    'WebRead',
]
