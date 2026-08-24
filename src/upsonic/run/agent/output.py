from __future__ import annotations

from dataclasses import dataclass, field
from time import time as current_time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Literal
)

from upsonic.run.base import RunStatus

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.messages.messages import (
        BinaryContent,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ThinkingPart,
    )
    from upsonic.run.agent.input import AgentRunInput
    from upsonic.run.events.events import AgentEvent
    from upsonic.run.requirements import RunRequirement
    from upsonic.run.tools.tools import ToolExecution
    from upsonic.usage import TaskUsage
    from upsonic.profiles import ModelProfile
    from upsonic.agent.pipeline.step import StepResult
    from upsonic.run.pipeline.stats import PipelineExecutionStats
    from upsonic.tasks.tasks import Task
    from upsonic.schemas.kb_filter import KBFilterExpr


@dataclass
class AgentRunOutput:
    """Complete output and runtime context for an agent run.
    
    This is the SINGLE SOURCE OF TRUTH for agent run state. It combines:
    - Runtime execution context
    - User-facing output and results
    
    Handles:
    - Message tracking with run boundaries
    - Chat history for LLM execution
    - Tool execution results and tracking
    - HITL requirements (external tool calls only)
    - Streaming state with async context manager
    - Event streaming
    - Text output streaming
    - Step execution tracking
    - Comprehensive serialization
    
    Attributes:
        run_id: Unique identifier for this run
        session_id: Session identifier
        user_id: User identifier
        task: Embedded task (single source of truth for HITL)
        step_results: List of StepResult tracking each step's execution
        execution_stats: Pipeline execution statistics
        requirements: HITL requirements for external tool calls
        agent_knowledge_base_filter: Vector search filters
        session_state: Session state persisted across runs
        output_schema: Output schema constraint
        is_streaming: Whether this is streaming execution
        accumulated_text: Text accumulated during streaming
        chat_history: Full conversation history (historical + current run messages) for LLM
        messages: Only NEW messages from THIS run
        response: Current ModelResponse (single, not list)
        output: Final processed output (str or bytes)
        events: All events emitted during execution
    """
    
    # --- Identity ---
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # --- Task (embedded for single source of truth) ---
    task: Optional["Task"] = None
    
    # Input reference
    input: Optional["AgentRunInput"] = None
    
    # --- Output ---
    output: Optional[Union[str, bytes]] = None  # Final agent output
    output_schema: Optional[Union[str, Type["BaseModel"]]] = None
    
    # Thinking/reasoning
    thinking_content: Optional[str] = None
    thinking_parts: Optional[List["ThinkingPart"]] = None
    
    # --- Model info ---
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    model_provider_profile: Optional["ModelProfile"] = None
    
    # --- Messages ---
    # chat_history: Full conversation history (historical + current) for LLM execution FOR THE SESSION, ALL RUNS
    chat_history: List["ModelMessage"] = field(default_factory=list)
    # messages: Only NEW messages from THIS run
    messages: Optional[List["ModelMessage"]] = None
    # response: Current ModelResponse
    response: Optional["ModelResponse"] = None
    usage: Optional["TaskUsage"] = None
    additional_input_message: Optional[List["ModelRequest"]] = None
    
    # Memory tracking
    memory_message_count: int = 0
    
    # --- Tool executions ---
    tools: Optional[List["ToolExecution"]] = None
    tool_call_count: int = 0
    guardrail_denied_tool_call_count: int = 0
    tool_limit_reached: bool = False
    
    # --- Media outputs ---
    images: Optional[List["BinaryContent"]] = None
    files: Optional[List["BinaryContent"]] = None
    
    # --- Status and HITL ---
    status: RunStatus = field(default_factory=lambda: RunStatus.running)
    requirements: Optional[List["RunRequirement"]] = None  # External tool requirements only
    step_results: List["StepResult"] = field(default_factory=list)

    # Pipeline execution statistics
    execution_stats: Optional["PipelineExecutionStats"] = None
    
    # --- Events (for streaming) ---
    events: Optional[List["AgentEvent"]] = None
    
    # --- Configuration ---
    agent_knowledge_base_filter: Optional["KBFilterExpr"] = None
    print_flag: bool = False  # Resolved print flag for this run (thread-safe)
    
    # --- Metadata ---
    metadata: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None

    # --- Skill metrics ---
    skill_metrics: Optional[Dict[str, Any]] = None

    
    # --- Execution state ---
    is_streaming: bool = False
    accumulated_text: str = ""
    
    # Current step result (set by Step.execute, read by Step.run)
    current_step_result: Optional["StepResult"] = None
    
    # --- User-facing pause information ---
    pause_reason: Optional[Literal["external_tool", "confirmation", "user_input"]] = None
    error_details: Optional[str] = None
    
    # --- Context management ---
    _context_window_full: bool = False
    
    # --- Message tracking (internal) ---
    # _run_boundaries tracks where each run starts in chat_history
    # This allows extracting new messages from just this run
    _run_boundaries: List[int] = field(default_factory=list)
    non_executed_tool_attempt_count: int = 0
    
    # --- Timestamps ---
    created_at: int = field(default_factory=lambda: int(current_time()))
    updated_at: Optional[int] = None
    
    
    @property
    def is_paused(self) -> bool:
        """Check if the run is paused (external tool execution)."""
        return self.status == RunStatus.paused
    
    @property
    def is_cancelled(self) -> bool:
        """Check if the run was cancelled."""
        return self.status == RunStatus.cancelled
    
    @property
    def is_complete(self) -> bool:
        """Check if the run is complete."""
        return self.status == RunStatus.completed
    
    @property
    def is_error(self) -> bool:
        """Check if the run has an error."""
        return self.status == RunStatus.error
    
    @property
    def is_problematic(self) -> bool:
        """Check if the run is problematic (paused, cancelled, or error) and needs continue_run_async."""
        return self.status in (RunStatus.paused, RunStatus.cancelled, RunStatus.error)
    
    @property
    def active_requirements(self) -> List["RunRequirement"]:
        """Get all unresolved requirements (confirmation, user input, or external execution)."""
        if not self.requirements:
            return []
        return [req for req in self.requirements if not req.is_resolved]
    
    @property
    def tools_requiring_confirmation(self) -> List["ToolExecution"]:
        """Get tools that require user confirmation."""
        tools_list: List["ToolExecution"] = []
        seen: set[str] = set()
        if self.tools:
            for t in self.tools:
                if t.requires_confirmation and t.confirmed is None:
                    if t.tool_call_id and t.tool_call_id not in seen:
                        tools_list.append(t)
                        seen.add(t.tool_call_id)
        if self.requirements:
            for req in self.requirements:
                if req.needs_confirmation and req.tool_execution:
                    te = req.tool_execution
                    if te.tool_call_id and te.tool_call_id not in seen:
                        tools_list.append(te)
                        seen.add(te.tool_call_id)
        return tools_list
    
    @property
    def tools_requiring_user_input(self) -> List["ToolExecution"]:
        """Get tools that require user input."""
        tools_list: List["ToolExecution"] = []
        seen: set[str] = set()
        if self.tools:
            for t in self.tools:
                if t.requires_user_input and not t.answered:
                    if t.tool_call_id and t.tool_call_id not in seen:
                        tools_list.append(t)
                        seen.add(t.tool_call_id)
        if self.requirements:
            for req in self.requirements:
                if req.needs_user_input and req.tool_execution:
                    te = req.tool_execution
                    if te.tool_call_id and te.tool_call_id not in seen:
                        tools_list.append(te)
                        seen.add(te.tool_call_id)
        return tools_list
    
    @property
    def tools_awaiting_external_execution(self) -> List["ToolExecution"]:
        """Get tools awaiting external execution."""
        tools_list: List["ToolExecution"] = []
        seen_tool_call_ids: set[str] = set()
        
        # Check tools directly
        if self.tools:
            for t in self.tools:
                if t.external_execution_required and t.result is None:
                    if t.tool_call_id and t.tool_call_id not in seen_tool_call_ids:
                        tools_list.append(t)
                        seen_tool_call_ids.add(t.tool_call_id)
                    elif not t.tool_call_id:
                        # If no tool_call_id, add anyway (shouldn't happen but be safe)
                        tools_list.append(t)
        
        # Also check requirements (external tools are stored here)
        if self.requirements:
            for req in self.requirements:
                if req.needs_external_execution and req.tool_execution:
                    tool_exec = req.tool_execution
                    if tool_exec.tool_call_id and tool_exec.tool_call_id not in seen_tool_call_ids:
                        tools_list.append(tool_exec)
                        seen_tool_call_ids.add(tool_exec.tool_call_id)
                    elif not tool_exec.tool_call_id:
                        # If no tool_call_id, add anyway (shouldn't happen but be safe)
                        tools_list.append(tool_exec)
        
        return tools_list
    
    # ========================================================================
    # Requirement Methods (HITL: External Tool, Confirmation, User Input)
    # ========================================================================
    
    def add_requirement(self, requirement: "RunRequirement") -> None:
        """Add an external tool requirement."""
        if self.requirements is None:
            self.requirements = []
        self.requirements.append(requirement)
    
    def get_external_tool_requirements(self) -> List["RunRequirement"]:
        """Get all external tool requirements."""
        if not self.requirements:
            return []
        return list(self.requirements)
    
    def get_external_tool_requirements_with_results(self) -> List["RunRequirement"]:
        """Get external tool requirements that have results."""
        if not self.requirements:
            return []
        return [
            r for r in self.requirements 
            if r.tool_execution and r.tool_execution.result is not None
        ]
    
    def has_pending_external_tools(self) -> bool:
        """Check if there are any external tools awaiting execution."""
        return len(self.active_requirements) > 0
    
    # ========================================================================
    # Step Result Methods
    # ========================================================================
    
    def get_step_results(self) -> List["StepResult"]:
        """Get step results from execution."""
        if self.step_results:
            return self.step_results
        return []
    
    def get_execution_stats(self) -> Optional["PipelineExecutionStats"]:
        """Get execution statistics."""
        return self.execution_stats
    
    def get_last_successful_step(self) -> Optional["StepResult"]:
        """Get the last successfully completed step."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in reversed(self.step_results):
            if result.status == StepStatus.COMPLETED:
                return result
        return None
    
    def get_error_step(self) -> Optional["StepResult"]:
        """Get the step that failed with ERROR status (for durable execution)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.ERROR:
                return result
        return None
    
    def get_cancelled_step(self) -> Optional["StepResult"]:
        """Get the step that was CANCELLED (for cancel run resumption)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.CANCELLED:
                return result
        return None
    
    def get_paused_step(self) -> Optional["StepResult"]:
        """Get the step that is PAUSED (for external tool execution)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.PAUSED:
                return result
        return None
    
    def get_problematic_step(self) -> Optional["StepResult"]:
        """Get the step that caused the run to be problematic (error, cancelled, or paused)."""
        return self.get_error_step() or self.get_cancelled_step() or self.get_paused_step()
    
    # ========================================================================
    # Message Tracking Methods
    # ========================================================================
    
    def start_new_run(self) -> None:
        """
        Mark the start of a new run.
        
        This records the current length of chat_history in _run_boundaries,
        allowing us to later extract only the new messages from this run.
        
        Should be called at the START of a run (in ChatHistoryStep) AFTER
        loading historical messages into chat_history.
        """
        # Record current chat_history length as the run boundary
        if self.chat_history:
            self._run_boundaries.append(len(self.chat_history))
        else:
            self._run_boundaries.append(0)
    
    def finalize_run_messages(self) -> None:
        """
        Finalize message tracking for this run.
        
        Extracts the NEW messages from chat_history (messages added after start_new_run
        was called) and stores them in self.messages.
        
        Should be called at the END of the run (in MemorySaveStep) AFTER all model
        interactions are complete (including steps like AgentPolicyStep that may
        add additional messages).
        
        Uses _run_boundaries to determine which messages are new:
        - chat_history[0:boundary] = historical messages from memory
        - chat_history[boundary:] = new messages from THIS run
        """
        if not self.chat_history:
            if self.messages is None:
                self.messages = []
            return
        
        # Get the run start boundary
        run_start = 0
        if self._run_boundaries:
            run_start = self._run_boundaries[-1]
        
        # Extract only the new messages added during this run
        if run_start < len(self.chat_history):
            new_messages = self.chat_history[run_start:]
            if self.messages is None:
                self.messages = []
            # Clear and set to avoid duplicates on multiple calls
            self.messages = list(new_messages)
        elif self.messages is None:
            self.messages = []
    
    def all_messages(self) -> List["ModelMessage"]:
        """Get all messages from this run."""
        return (self.messages or []).copy()
    
    def new_messages(self) -> List["ModelMessage"]:
        """
        Get only the NEW messages from this run.
        
        Returns messages that were added during this specific run,
        excluding any historical messages loaded from memory.
        
        For proper tracking:
        - start_new_run() should be called at run start
        - finalize_run_messages() should be called at run end
        
        Returns:
            List of ModelMessage added during this run only.
        """
        # If messages was set via finalize_run_messages, return it
        if self.messages:
            return self.messages.copy()
        
        # Fallback: Calculate from _run_boundaries and chat_history
        if self._run_boundaries and self.chat_history:
            run_start = self._run_boundaries[-1]
            if run_start < len(self.chat_history):
                return self.chat_history[run_start:].copy()
        
        # No tracking info available, return empty
        return []
    
    def add_message(self, message: "ModelMessage") -> None:
        """Add a single message to the run's message list."""
        if self.messages is None:
            self.messages = []
        self.messages.append(message)
    
    def add_messages(self, messages: List["ModelMessage"]) -> None:
        """Add multiple messages to the run's message list.
        
        This is a batch version of add_message for convenience.
        
        Args:
            messages: List of ModelMessage objects to add.
        """
        if self.messages is None:
            self.messages = []
        self.messages.extend(messages)
    
    def get_last_model_response(self) -> Optional["ModelResponse"]:
        """Get the last ModelResponse from the run's messages."""
        from upsonic.messages.messages import ModelResponse
        
        messages = self.new_messages()
        for msg in reversed(messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None
    
    def has_new_messages(self) -> bool:
        """Check if this run has any new messages."""
        if self.messages:
            return True
        # Check via boundaries
        if self._run_boundaries and self.chat_history:
            run_start = self._run_boundaries[-1]
            return run_start < len(self.chat_history)
        return False
    
    # ========================================================================
    # Status Methods
    # ========================================================================
    
    def _sync_status_to_task(self) -> None:
        """Sync the current status to the embedded task."""
        if self.task is not None:
            self.task.status = self.status
    
    def mark_paused(self, reason: Literal["external_tool", "confirmation", "user_input"] = "external_tool") -> None:
        """Mark the run as paused for HITL interaction."""
        self.status = RunStatus.paused
        self.pause_reason = reason
        self.updated_at = int(current_time())
        self._sync_status_to_task()
    
    def mark_cancelled(self) -> None:
        """Mark the run as cancelled."""
        self.status = RunStatus.cancelled
        self.updated_at = int(current_time())
        self._sync_status_to_task()
    
    def mark_completed(self) -> None:
        """Mark the run as completed and finalize output."""
        self.status = RunStatus.completed
        self.updated_at = int(current_time())
        # Set output from accumulated_text if streaming
        if self.output is None and self.accumulated_text and self.is_streaming:
            self.output = self.accumulated_text
        self._sync_status_to_task()
    
    def mark_error(self, error: Optional[str] = None) -> None:
        """Mark the run as having an error."""
        self.status = RunStatus.error
        if error:
            self.error_details = error
            if self.metadata is None:
                self.metadata = {}
            self.metadata["error"] = error
        self.updated_at = int(current_time())
        self._sync_status_to_task()
    
    # ========================================================================
    # Usage Tracking Methods
    # ========================================================================
    
    def _ensure_usage(self) -> "TaskUsage":
        """Ensure usage object exists and return it.

        Handles conversion from dict (from storage deserialization) to TaskUsage.
        """
        from upsonic.usage import TaskUsage

        if self.usage is None:
            self.usage = TaskUsage()
        elif isinstance(self.usage, dict):
            self.usage = TaskUsage.from_dict(self.usage)

        return self.usage
    
    def start_usage_timer(self) -> None:
        """Start the usage timer for this run.
        
        Should be called at the beginning of run execution.
        """
        usage = self._ensure_usage()
        usage.start_timer()
    
    def stop_usage_timer(self, set_duration: bool = True) -> None:
        """Stop the usage timer and optionally set duration.
        
        Should be called at the end of run execution.
        
        Args:
            set_duration: If True, set duration from timer.elapsed
        """
        if self.usage is not None:
            self.usage.stop_timer(set_duration=set_duration)
    
    def set_usage_time_to_first_token(self) -> None:
        """Record time to first token from the usage timer.
        
        Should be called when the first token is generated during streaming.
        """
        if self.usage is not None:
            self.usage.set_time_to_first_token()
    
    def increment_tool_calls(self, count: int = 1) -> None:
        """Increment the tool call count in usage.
        
        Args:
            count: Number of tool calls to add.
        """
        usage = self._ensure_usage()
        usage.tool_calls += count
    
    def add_model_execution_time(self, elapsed: float) -> None:
        """Accumulate model (LLM API) execution time into usage.

        Args:
            elapsed: Time in seconds spent in a single model.request() call.
        """
        usage = self._ensure_usage()
        usage.add_model_execution_time(elapsed)

    def add_tool_execution_time(self, elapsed: float) -> None:
        """Accumulate tool execution time into usage.

        Args:
            elapsed: Time in seconds spent in a single tool execution.
        """
        usage = self._ensure_usage()
        usage.add_tool_execution_time(elapsed)

    def set_usage_cost(self, cost: float) -> None:
        """Set the cost in usage.
        
        Args:
            cost: The cost value to set.
        """
        usage = self._ensure_usage()
        if usage.cost is None:
            usage.cost = cost
        else:
            usage.cost += cost
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    def to_dict(self, serialize_flag: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            serialize_flag: Passed to Task.to_dict for cloudpickle serialization.
        
        Uses TypeAdapters for proper serialization:
        - chat_history/messages/response/additional_input_message: ModelMessagesTypeAdapter
        - thinking_parts: ModelResponsePartTypeAdapter
        - images/files: BinaryContentTypeAdapter
        - usage: dataclasses.asdict
        - tools: to_dict method
        - status: to_dict method
        - requirements: to_dict method
        - step_results: to_dict method
        - execution_stats: to_dict method
        - events: to_dict method
        - agent_knowledge_base_filter: to_dict method
        - current_step_result: to_dict method
        - output_schema: type handling (str or BaseModel class name/module)
        """
        from pydantic import BaseModel
        from upsonic.messages.messages import (
            BinaryContentTypeAdapter,
            ModelMessagesTypeAdapter,
            ModelResponsePartTypeAdapter,
        )
        
        result: Dict[str, Any] = {
            # Simple/primitive attributes
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "user_id": self.user_id,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "memory_message_count": self.memory_message_count,
            "tool_call_count": self.tool_call_count,
            "guardrail_denied_tool_call_count": self.guardrail_denied_tool_call_count,
            "non_executed_tool_attempt_count": self.non_executed_tool_attempt_count,
            "tool_limit_reached": self.tool_limit_reached,
            "metadata": self.metadata,
            "session_state": self.session_state,
            "is_streaming": self.is_streaming,
            "accumulated_text": self.accumulated_text,
            "_run_boundaries": self._run_boundaries,
            "pause_reason": self.pause_reason,
            "error_details": self.error_details,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "output": self.output,
            "thinking_content": self.thinking_content,
            "_context_window_full": self._context_window_full,
            "print_flag": self.print_flag,
            "trace_id": self.trace_id,
        }
        
        # task: use to_dict with serialize_flag
        if self.task is not None:
            result["task"] = self.task.to_dict(serialize_flag=serialize_flag)
        else:
            result["task"] = None
        
        # input: AgentRunInput.to_dict()
        if self.input is not None:
            result["input"] = self.input.to_dict()
        else:
            result["input"] = None
        
        # output_schema: type handling (leave as-is if type)
        if self.output_schema is None:
            result["output_schema"] = None
        elif self.output_schema is str:
            result["output_schema"] = {"__builtin_type__": True, "name": "str"}
        elif isinstance(self.output_schema, type):
            try:
                if issubclass(self.output_schema, BaseModel):
                    result["output_schema"] = {
                        "__pydantic_type__": True,
                        "name": self.output_schema.__name__,
                        "module": self.output_schema.__module__,
                    }
                else:
                    result["output_schema"] = {
                        "__type__": True,
                        "name": self.output_schema.__name__,
                        "module": self.output_schema.__module__,
                    }
            except TypeError:
                result["output_schema"] = str(self.output_schema)
        else:
            result["output_schema"] = str(self.output_schema)
        
        # thinking_parts: ModelResponsePartTypeAdapter
        if self.thinking_parts:
            result["thinking_parts"] = ModelResponsePartTypeAdapter.dump_python(
                self.thinking_parts, mode="json"
            )
        else:
            result["thinking_parts"] = None
        
        # model_provider_profile: to_dict
        if self.model_provider_profile is not None:
            result["model_provider_profile"] = self.model_provider_profile.to_dict()
        else:
            result["model_provider_profile"] = None
        
        # chat_history: ModelMessagesTypeAdapter
        if self.chat_history:
            result["chat_history"] = ModelMessagesTypeAdapter.dump_python(
                self.chat_history, mode="json"
            )
        else:
            result["chat_history"] = []
        
        # messages: ModelMessagesTypeAdapter
        if self.messages:
            result["messages"] = ModelMessagesTypeAdapter.dump_python(
                self.messages, mode="json"
            )
        else:
            result["messages"] = None
        
        # response: ModelMessagesTypeAdapter (single item in list)
        if self.response is not None:
            result["response"] = ModelMessagesTypeAdapter.dump_python(
                [self.response], mode="json"
            )[0]
        else:
            result["response"] = None
        
        # usage: RunUsage.to_dict()
        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        else:
            result["usage"] = None

        # additional_input_message: ModelMessagesTypeAdapter
        if self.additional_input_message:
            result["additional_input_message"] = ModelMessagesTypeAdapter.dump_python(
                self.additional_input_message, mode="json"
            )
        else:
            result["additional_input_message"] = None
        
        # tools: to_dict
        if self.tools:
            result["tools"] = [t.to_dict() for t in self.tools]
        else:
            result["tools"] = None
        
        # images: BinaryContentTypeAdapter
        if self.images:
            result["images"] = BinaryContentTypeAdapter.dump_python(
                self.images, mode="json"
            )
        else:
            result["images"] = None
        
        # files: BinaryContentTypeAdapter
        if self.files:
            result["files"] = BinaryContentTypeAdapter.dump_python(
                self.files, mode="json"
            )
        else:
            result["files"] = None
        
        # status: to_dict
        if self.status is not None:
            result["status"] = self.status.to_dict()
        else:
            result["status"] = None
        
        # requirements: to_dict
        if self.requirements:
            result["requirements"] = [r.to_dict() for r in self.requirements]
        else:
            result["requirements"] = None
        
        # step_results: to_dict
        if self.step_results:
            result["step_results"] = [sr.to_dict() for sr in self.step_results]
        else:
            result["step_results"] = []
        
        # execution_stats: to_dict
        if self.execution_stats is not None:
            result["execution_stats"] = self.execution_stats.to_dict()
        else:
            result["execution_stats"] = None
        
        # events: to_dict
        if self.events:
            result["events"] = [e.to_dict() for e in self.events]
        else:
            result["events"] = None
        
        # skill_metrics: dict of {skill_name: SkillMetrics.to_dict()}
        if self.skill_metrics:
            result["skill_metrics"] = self.skill_metrics
        else:
            result["skill_metrics"] = None

        # agent_knowledge_base_filter: to_dict
        if self.agent_knowledge_base_filter is not None:
            result["agent_knowledge_base_filter"] = self.agent_knowledge_base_filter.to_dict()
        else:
            result["agent_knowledge_base_filter"] = None
        
        # current_step_result: to_dict
        if self.current_step_result is not None:
            result["current_step_result"] = self.current_step_result.to_dict()
        else:
            result["current_step_result"] = None
        
        return result
    
    def to_json(self, indent: Optional[int] = 2, serialize_flag: bool = False) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(serialize_flag=serialize_flag), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], deserialize_flag: bool = False) -> "AgentRunOutput":
        """
        Reconstruct from dictionary.
        
        Args:
            data: Dictionary containing AgentRunOutput data
            deserialize_flag: Passed to Task.from_dict for cloudpickle deserialization.
        
        Uses TypeAdapters for proper deserialization:
        - chat_history/messages/response/additional_input_message: ModelMessagesTypeAdapter
        - thinking_parts: ModelResponsePartTypeAdapter
        - images/files: BinaryContentTypeAdapter
        - usage: RequestUsage constructor
        - output_schema: type handling (str or BaseModel class)
        """
        import importlib
        from upsonic.messages.messages import (
            BinaryContentTypeAdapter,
            ModelMessagesTypeAdapter,
            ModelResponsePartTypeAdapter,
        )
        from upsonic.run.agent.input import AgentRunInput
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution
        from upsonic.run.events.events import AgentEvent
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        from upsonic.agent.pipeline.step import StepResult
        from upsonic.tasks.tasks import Task
        from upsonic.profiles import ModelProfile
        from upsonic.usage import TaskUsage
        from upsonic.schemas.kb_filter import KBFilterExpr

        # Handle task (dict or Task object) - with deserialize_flag
        task = None
        task_data = data.get("task")
        if isinstance(task_data, dict):
            task = Task.from_dict(task_data, deserialize_flag=deserialize_flag)
        else:
            task = task_data
        
        # Handle input (dict or object)
        input_data = data.get("input")
        input_obj = None
        if isinstance(input_data, dict):
            input_obj = AgentRunInput.from_dict(input_data)
        else:
            input_obj = input_data
        
        # Handle output_schema (type handling)
        output_schema_data = data.get("output_schema")
        output_schema: Optional[Union[str, Type["BaseModel"]]] = None
        if output_schema_data is None:
            output_schema = None
        elif isinstance(output_schema_data, dict):
            if output_schema_data.get("__builtin_type__") and output_schema_data.get("name") == "str":
                output_schema = str
            elif output_schema_data.get("__pydantic_type__"):
                module_name = output_schema_data.get("module")
                class_name = output_schema_data.get("name")
                if module_name and class_name:
                    try:
                        module = importlib.import_module(module_name)
                        output_schema = getattr(module, class_name)
                    except (ImportError, AttributeError):
                        output_schema = None
            elif output_schema_data.get("__type__"):
                module_name = output_schema_data.get("module")
                class_name = output_schema_data.get("name")
                if module_name and class_name:
                    try:
                        module = importlib.import_module(module_name)
                        output_schema = getattr(module, class_name)
                    except (ImportError, AttributeError):
                        output_schema = None
        elif isinstance(output_schema_data, type):
            output_schema = output_schema_data
        else:
            output_schema = output_schema_data
        
        # Handle thinking_parts: ModelResponsePartTypeAdapter
        thinking_parts_data = data.get("thinking_parts")
        if thinking_parts_data and isinstance(thinking_parts_data, list):
            thinking_parts = ModelResponsePartTypeAdapter.validate_python(thinking_parts_data)
        else:
            thinking_parts = None
        
        # Handle model_provider_profile (dict or ModelProfile)
        model_provider_profile = None
        model_provider_profile_data = data.get("model_provider_profile")
        if isinstance(model_provider_profile_data, dict):
            model_provider_profile = ModelProfile.from_dict(model_provider_profile_data)
        else:
            model_provider_profile = model_provider_profile_data
        
        # Handle chat_history: ModelMessagesTypeAdapter
        chat_history_data = data.get("chat_history", [])
        if chat_history_data and isinstance(chat_history_data, list):
            chat_history = ModelMessagesTypeAdapter.validate_python(chat_history_data)
        else:
            chat_history = []
        
        # Handle messages: ModelMessagesTypeAdapter
        messages_data = data.get("messages")
        if messages_data and isinstance(messages_data, list):
            messages = ModelMessagesTypeAdapter.validate_python(messages_data)
        else:
            messages = None
        
        # Handle response: ModelMessagesTypeAdapter (single item)
        response_data = data.get("response")
        if response_data and isinstance(response_data, dict):
            response_list = ModelMessagesTypeAdapter.validate_python([response_data])
            response = response_list[0] if response_list else None
        else:
            response = response_data
        
        usage_data = data.get("usage")
        if usage_data and isinstance(usage_data, dict):
            usage = TaskUsage.from_dict(usage_data)
        else:
            usage = usage_data

        # Handle additional_input_message: ModelMessagesTypeAdapter
        additional_input_message_data = data.get("additional_input_message")
        if additional_input_message_data and isinstance(additional_input_message_data, list):
            additional_input_message = ModelMessagesTypeAdapter.validate_python(additional_input_message_data)
        else:
            additional_input_message = None
        
        # Handle tools (list of dicts or objects)
        tools = None
        tools_data = data.get("tools")
        if tools_data:
            tools = []
            for t in tools_data:
                if isinstance(t, dict):
                    tools.append(ToolExecution.from_dict(t))
                else:
                    tools.append(t)
        
        # Handle images: BinaryContentTypeAdapter
        images_data = data.get("images")
        if images_data and isinstance(images_data, list):
            images = BinaryContentTypeAdapter.validate_python(images_data)
        else:
            images = None
        
        # Handle files: BinaryContentTypeAdapter
        files_data = data.get("files")
        if files_data and isinstance(files_data, list):
            files = BinaryContentTypeAdapter.validate_python(files_data)
        else:
            files = None
        
        # Handle status: from_dict
        status_data = data.get("status", RunStatus.running.value)
        if isinstance(status_data, str):
            status = RunStatus.from_dict(status_data)
        elif isinstance(status_data, RunStatus):
            status = status_data
        else:
            status = RunStatus.running
        
        # Handle requirements (list of dicts or objects)
        requirements = None
        requirements_data = data.get("requirements")
        if requirements_data:
            requirements = []
            for r in requirements_data:
                if isinstance(r, dict):
                    requirements.append(RunRequirement.from_dict(r))
                else:
                    requirements.append(r)
        
        # Handle step_results (list of dicts)
        step_results: List[Any] = []
        step_results_data = data.get("step_results", [])
        for sr in step_results_data:
            if isinstance(sr, dict):
                step_results.append(StepResult.from_dict(sr))
            else:
                step_results.append(sr)

        # Handle execution_stats (dict)
        execution_stats = None
        execution_stats_data = data.get("execution_stats")
        if isinstance(execution_stats_data, dict):
            execution_stats = PipelineExecutionStats.from_dict(execution_stats_data)
        else:
            execution_stats = execution_stats_data
        
        # Handle events (list of dicts or objects)
        events = None
        events_data = data.get("events")
        if events_data:
            events = []
            for e in events_data:
                if isinstance(e, dict):
                    events.append(AgentEvent.from_dict(e))
                else:
                    events.append(e)
        
        # Handle agent_knowledge_base_filter: from_dict
        agent_knowledge_base_filter_data = data.get("agent_knowledge_base_filter")
        if agent_knowledge_base_filter_data and isinstance(agent_knowledge_base_filter_data, dict):
            agent_knowledge_base_filter = KBFilterExpr.from_dict(agent_knowledge_base_filter_data)
        else:
            agent_knowledge_base_filter = agent_knowledge_base_filter_data
        
        # Handle skill_metrics (already plain dict)
        skill_metrics = data.get("skill_metrics")

        # Handle current_step_result: from_dict
        current_step_result_data = data.get("current_step_result")
        if current_step_result_data and isinstance(current_step_result_data, dict):
            current_step_result = StepResult.from_dict(current_step_result_data)
        else:
            current_step_result = current_step_result_data
        
        
        return cls(
            run_id=data.get("run_id"),
            agent_id=data.get("agent_id"),
            agent_name=data.get("agent_name"),
            session_id=data.get("session_id"),
            parent_run_id=data.get("parent_run_id"),
            user_id=data.get("user_id"),
            trace_id=data.get("trace_id"),
            task=task,
            status=status,
            created_at=data.get("created_at", int(current_time())),
            updated_at=data.get("updated_at"),
            input=input_obj,
            output=data.get("output"),
            output_schema=output_schema,
            thinking_content=data.get("thinking_content"),
            thinking_parts=thinking_parts,
            model_name=data.get("model_name"),
            model_provider=data.get("model_provider"),
            model_provider_profile=model_provider_profile,
            chat_history=chat_history,
            messages=messages,
            response=response,
            usage=usage,
            additional_input_message=additional_input_message,
            memory_message_count=data.get("memory_message_count", 0),
            tools=tools,
            tool_call_count=data.get("tool_call_count", 0),
            guardrail_denied_tool_call_count=data.get("guardrail_denied_tool_call_count", 0),
            non_executed_tool_attempt_count=data.get("non_executed_tool_attempt_count", 0),
            tool_limit_reached=data.get("tool_limit_reached", False),
            images=images,
            files=files,
            requirements=requirements,
            step_results=step_results,
            execution_stats=execution_stats,
            events=events,
            agent_knowledge_base_filter=agent_knowledge_base_filter,
            metadata=data.get("metadata"),
            session_state=data.get("session_state"),
            skill_metrics=skill_metrics,
            is_streaming=data.get("is_streaming", False),
            accumulated_text=data.get("accumulated_text", ""),
            current_step_result=current_step_result,
            _run_boundaries=data.get("_run_boundaries", []),
            pause_reason=data.get("pause_reason"),
            error_details=data.get("error_details"),
            _context_window_full=data.get("_context_window_full", False),
            print_flag=data.get("print_flag", False),
        )
    
    def __str__(self) -> str:
        return str(self.output if self.output is not None else "")
    
    def __repr__(self) -> str:
        return f"AgentRunOutput(run_id={self.run_id!r}, status={self.status.value}, output_length={len(str(self.output or ''))})"
