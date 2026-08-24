"""
Pipeline Manager - Orchestrates step execution.

The PipelineManager is responsible for executing a sequence of steps
in order, managing the context flow, and handling the overall execution.
It holds references to task, agent, and model, passing them to each step.
It also supports comprehensive event streaming for full visibility into
the execution pipeline.
"""

import time
from typing import List, Optional, Dict, Any, AsyncIterator, TYPE_CHECKING
from .step import Step, StepStatus

if TYPE_CHECKING:
    from upsonic.run.agent.output import AgentRunOutput
    from upsonic.agent.pipeline.step import StepResult
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model
    from upsonic.agent.agent import Agent
    from upsonic.tools.hitl import ExternalExecutionPause, ConfirmationPause, UserInputPause
    from upsonic.run.events.events import AgentEvent
    from upsonic.agent.otel_manager import AgentOTelManager
else:
    AgentRunOutput = "AgentRunOutput"
    StepResult = "StepResult"
    Task = "Task"
    Model = "Model"
    Agent = "Agent"
    AgentEvent = "AgentEvent"
    AgentOTelManager = "AgentOTelManager"
    ExternalExecutionPause = "ExternalExecutionPause"
    ConfirmationPause = "ConfirmationPause"
    UserInputPause = "UserInputPause"

from upsonic.utils.logging_config import sentry_sdk, get_logger

_sentry_logger = get_logger("upsonic.sentry.pipeline")


class PipelineManager:
    """
    Manages the execution of a pipeline of steps.
    
    The PipelineManager orchestrates the agent execution by:
    - Running steps in sequence
    - Managing the shared context
    - Handling errors and early termination
    - Providing execution statistics
    - Emitting events for streaming visibility
    """
    
    def __init__(
        self,
        steps: Optional[List[Step]] = None,
        task: Optional["Task"] = None,
        agent: Optional["Agent"] = None,
        model: Optional["Model"] = None,
        debug: bool = False
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            steps: List of steps to execute
            task: The task being executed
            agent: The agent instance
            model: The model being used
            debug: Enable debug logging
        """
        self.steps: List[Step] = steps or []
        self.task = task
        self.agent = agent
        self.model = model
        self.debug = debug
        
        # Manager registry for sharing managers across steps
        self._managers: Dict[str, Any] = {}
    
    @property
    def _otel(self) -> "AgentOTelManager":
        """Resolve the AgentOTelManager from the agent, with a no-op fallback."""
        if self.agent is not None:
            otel = getattr(self.agent, "_otel", None)
            if otel is not None:
                return otel
        from upsonic.agent.otel_manager import AgentOTelManager as _Mgr
        return _Mgr(None)

    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)
    
    def insert_step(self, index: int, step: Step) -> None:
        """Insert a step at a specific position."""
        self.steps.insert(index, step)
    
    def remove_step(self, step_name: str) -> bool:
        """Remove a step by name."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                self.steps.pop(i)
                return True
        return False
    
    def get_step(self, step_name: str) -> Optional[Step]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    async def _save_session(self, output: "AgentRunOutput") -> None:
        """
        Save session to storage via Memory class.
        
        This is the centralized session saving for ALL scenarios:
        - Successful completion (with memory features like summary, user profile)
        - External tool pauses (checkpoint only)
        - Cancel run (checkpoint only)
        - Durable execution / error recovery (checkpoint only)
        
        Args:
            output: The agent run output to save
        """
        if not self.agent:
            return
        
        if self.agent.memory:
            try:
                await self.agent.memory.save_session_async(
                    output=output,
                    agent_id=self.agent.agent_id,
                )
            except Exception as save_error:
                if self.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Failed to save session: {save_error}", "PipelineManager")
    
    def _handle_cancellation(
        self,
        output: "AgentRunOutput",
        cancelled_step_result: Optional["StepResult"] = None
    ) -> None:
        """
        Handle run cancellation by marking output as cancelled.
        
        No requirements are created - the run status and step_results contain all needed info
        for resumption via continue_run_async.
        
        Args:
            output: The agent run output (single source of truth)
            cancelled_step_result: Optional cancelled step result from output
        """
        if not cancelled_step_result:
            cancelled_step_result = output.get_cancelled_step()

        # Stop the usage timer and mark paused so the run-state knows
        # to resume cleanly; duration is preserved for the resume path.
        if self.task:
            self.task.is_paused = True
            if hasattr(self.task, '_usage') and self.task._usage is not None:
                self.task._usage.stop_timer()

        output.mark_cancelled()

        if self.debug:
            from upsonic.utils.printing import warning_log
            step_name = cancelled_step_result.name if cancelled_step_result else 'unknown'
            warning_log(f"Run cancelled at step {step_name}", "PipelineManager")
    
    async def _ahandle_cancellation(
        self,
        output: "AgentRunOutput",
        cancelled_step_result: Optional["StepResult"] = None
    ) -> None:
        """
        Async version of _handle_cancellation.
        
        Args:
            output: The agent run output (single source of truth)
            cancelled_step_result: Optional cancelled step result from output
        """
        self._handle_cancellation(output, cancelled_step_result)
        await self._save_session(output)
    
    def _handle_durable_execution_error(
        self,
        output: "AgentRunOutput",
        failed_step_result: Optional["StepResult"] = None
    ) -> None:
        """
        Handle durable execution error by marking output as error.
        
        No requirements are created - the run status and step_results contain all needed info
        for resumption via continue_run_async.
        
        Args:
            output: The agent run output (single source of truth)
            failed_step_result: Optional failed step result from output
        """
        if not failed_step_result:
            failed_step_result = output.get_error_step()

        # Stop the usage timer and mark paused so the run-state knows
        # to resume cleanly; duration is preserved for the resume path.
        if self.task:
            self.task.is_paused = True
            if hasattr(self.task, '_usage') and self.task._usage is not None:
                self.task._usage.stop_timer()

        error_msg = failed_step_result.message if failed_step_result else None
        output.mark_error(error_msg)
    
    async def _ahandle_durable_execution_error(
        self,
        output: "AgentRunOutput",
        failed_step_result: Optional["StepResult"] = None
    ) -> None:
        """
        Async version of _handle_durable_execution_error.
        
        Args:
            output: The agent run output (single source of truth)
            failed_step_result: Optional failed step result from output
        """
        self._handle_durable_execution_error(output, failed_step_result)
        await self._save_session(output)
    
    async def execute(
        self, 
        context: "AgentRunOutput",
        start_step_index: int = 0
    ) -> "AgentRunOutput":
        """
        Execute the pipeline (non-streaming).

        Runs all steps in sequence, passing the output through each one.
        All steps must execute. If any step raises an error, the pipeline stops,
        logs the error properly, and raises it to the caller.

        Args:
            context: The agent run output (single source of truth)
            start_step_index: Index to start execution from (0-based). Used for HITL resumption.

        Returns:
            AgentRunOutput: The final output after all steps

        Raises:
            Exception: Any exception from step execution is raised with proper error message
        """
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        if not context.execution_stats:
            context.execution_stats = PipelineExecutionStats(total_steps=len(self.steps))
        
        if start_step_index > 0:
            context.execution_stats.resumed_from = start_step_index
            if self.debug:
                from upsonic.utils.printing import info_log
                step_name = self.steps[start_step_index].name if start_step_index < len(self.steps) else "unknown"
                info_log(f"Resuming pipeline from step {start_step_index} ({step_name})", "PipelineManager")
        
        with self._otel.pipeline_span(
            total_steps=len(self.steps),
            is_streaming=context.is_streaming if hasattr(context, 'is_streaming') else False,
            debug=self.debug,
        ) as otel_pipeline_span, sentry_sdk.start_transaction(
            op="agent.pipeline.execute",
            name=f"Agent Pipeline ({len(self.steps)} steps)"
        ) as transaction:
            transaction.set_tag("pipeline.total_steps", len(self.steps))
            transaction.set_tag("pipeline.debug", self.debug)
            transaction.set_tag("pipeline.streaming", context.is_streaming if hasattr(context, 'is_streaming') else False)

            if self.task:
                transaction.set_tag("task.type", type(self.task).__name__)
                if hasattr(self.task, 'description'):
                    transaction.set_data("task.description", str(self.task.description)[:200])

            _sentry_logger.info(
                "Pipeline started: %d steps",
                len(self.steps),
                extra={
                    "total_steps": len(self.steps),
                    "debug": self.debug,
                    "streaming": context.is_streaming if hasattr(context, 'is_streaming') else False
                }
            )

            if self.debug:
                from upsonic.utils.printing import pipeline_started
                pipeline_started(len(self.steps))

            try:
                for step_index in range(start_step_index, len(self.steps)):
                    step = self.steps[step_index]
                    
                    with self._otel.step_span(step.name, step.description) as otel_step_span, sentry_sdk.start_span(
                        op=f"pipeline.step.{step.name}",
                        name=step.description
                    ) as span:
                        span.set_tag("step.name", step.name)
                        span.set_data("step.description", step.description)

                        if self.debug:
                            from upsonic.utils.printing import pipeline_step_started, debug_log_level2
                            pipeline_step_started(step.name, step.description)
                            
                            debug_level = getattr(self.agent, 'debug_level', 1) if self.agent else 1
                            if debug_level >= 2:
                                debug_log_level2(
                                    f"Pipeline step starting: {step.name}",
                                    "PipelineManager",
                                    debug=self.debug,
                                    debug_level=debug_level,
                                    step_name=step.name,
                                    step_description=step.description,
                                    step_index=step_index,
                                    total_steps=len(self.steps),
                                    task_description=self.task.description[:200] if self.task else None,
                                    agent_name=getattr(self.agent, 'name', 'Unknown') if self.agent else None,
                                    model_name=getattr(self.model, 'model_name', 'Unknown') if self.model else None,
                                    is_streaming=context.is_streaming
                                )

                        result = await step.run(context, self.task, self.agent, self.model, step_index, pipeline_manager=self)
                        
                        self._otel.set_step_result(
                            otel_step_span,
                            status=result.status.value,
                            execution_time=result.execution_time,
                            error_message=result.message if result.status == StepStatus.ERROR else None,
                        )

                        span.set_tag("step.status", result.status.value)
                        span.set_data("step.message", result.message)
                        span.set_data("step.execution_time", result.execution_time)

                        if self.debug:
                            from upsonic.utils.printing import pipeline_step_completed, debug_log_level2
                            pipeline_step_completed(
                                step.name,
                                result.status.value,
                                result.execution_time,
                                result.message
                            )
                            
                            debug_level = getattr(self.agent, 'debug_level', 1) if self.agent else 1
                            if debug_level >= 2:
                                debug_log_level2(
                                    f"Pipeline step completed: {step.name}",
                                    "PipelineManager",
                                    debug=self.debug,
                                    debug_level=debug_level,
                                    step_name=step.name,
                                    step_status=result.status.value,
                                    execution_time=result.execution_time,
                                    step_message=result.message,
                                    cumulative_execution_time=sum(context.execution_stats.step_timing.values()),
                                    steps_completed=context.execution_stats.executed_steps
                                )


                total_time = sum(context.execution_stats.step_timing.values())
                
                transaction.set_tag("pipeline.status", "success")
                transaction.set_data("pipeline.executed_steps", context.execution_stats.executed_steps)
                transaction.set_data("pipeline.total_time", total_time)

                _sentry_logger.info(
                    "Pipeline completed: %d/%d steps, %.3fs",
                    context.execution_stats.executed_steps,
                    len(self.steps),
                    total_time,
                    extra={
                        "executed_steps": context.execution_stats.executed_steps,
                        "total_steps": len(self.steps),
                        "total_time": total_time,
                        "status": "success"
                    }
                )

                if self.debug:
                    from upsonic.utils.printing import pipeline_completed, pipeline_timeline
                    pipeline_completed(
                        context.execution_stats.executed_steps,
                        len(self.steps),
                        total_time
                    )
                    step_results_dict = {
                        sr.name: {
                            "status": sr.status.value,
                            "message": sr.message,
                            "execution_time": sr.execution_time
                        }
                        for sr in context.step_results
                    }
                    pipeline_timeline(
                        step_results_dict,
                        total_time
                    )
                self._otel.mark_success(otel_pipeline_span)
                return context

            except Exception as e:
                from upsonic.exceptions import RunCancelledException
                from upsonic.tools.hitl import ExternalExecutionPause, ConfirmationPause, UserInputPause
                
                # Confirmation pause - return normally with paused status
                if isinstance(e, ConfirmationPause):
                    await self._handle_confirmation_pause(context, e)
                    return context

                # User input pause - return normally with paused status
                if isinstance(e, UserInputPause):
                    await self._handle_user_input_pause(context, e)
                    return context

                # External tool pause - return normally with paused status
                if isinstance(e, ExternalExecutionPause):
                    await self._handle_external_tool_pause(context, e)
                    return context
                
                # Cancel run - return normally with cancelled status (don't re-raise)
                if isinstance(e, RunCancelledException):
                    await self._ahandle_cancellation(context)
                    return context
                
                # Durable execution (error recovery) - save checkpoint and re-raise
                failed_step_result = context.get_error_step()
                await self._ahandle_durable_execution_error(context, failed_step_result)
                
                if self.debug:
                    from upsonic.utils.printing import warning_log, pipeline_failed, debug_log_level2
                    import traceback
                    step_name = failed_step_result.name if failed_step_result else 'unknown'
                    step_number = failed_step_result.step_number if failed_step_result else 0
                    warning_log(
                        f"❌ ERROR at step {step_number} ({step_name}): {str(e)[:100]}",
                        "PipelineManager"
                    )
                    executed_steps = context.execution_stats.executed_steps if context.execution_stats else 0
                    total_steps = context.execution_stats.total_steps if context.execution_stats else len(self.steps)
                    pipeline_failed(
                        str(e),
                        executed_steps,
                        total_steps,
                        failed_step_result.message if failed_step_result else None,
                        failed_step_result.execution_time if failed_step_result else None
                    )
                    
                    debug_level = getattr(self.agent, 'debug_level', 1) if self.agent else 1
                    if debug_level >= 2:
                        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        step_results_dict = {
                            sr.name: {
                                "status": sr.status.value,
                                "message": sr.message,
                                "execution_time": sr.execution_time
                            }
                            for sr in context.step_results
                        }
                        debug_log_level2(
                            "Pipeline execution error",
                            "PipelineManager",
                            debug=self.debug,
                            debug_level=debug_level,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            error_traceback=error_traceback[-2000:],
                            failed_step_index=failed_step_result.step_number if failed_step_result else 0,
                            failed_step_name=failed_step_result.name if failed_step_result else "Unknown",
                            executed_steps=executed_steps,
                            total_steps=total_steps,
                            step_results=step_results_dict,
                            task_description=self.task.description[:300] if self.task else None,
                            agent_name=getattr(self.agent, 'name', 'Unknown') if self.agent else None,
                            model_name=getattr(self.model, 'model_name', 'Unknown') if self.model else None
                        )
                
                self._otel.record_error(otel_pipeline_span, e)

                transaction.set_tag("pipeline.status", "error")
                transaction.set_data("error.message", str(e))
                transaction.set_data("error.type", type(e).__name__)

                sentry_sdk.capture_exception(e)
                raise
    
    async def execute_stream(
        self, 
        context: "AgentRunOutput",
        start_step_index: int = 0
    ) -> AsyncIterator["AgentEvent"]:
        """
        Execute the pipeline in streaming mode with comprehensive event emission.
        
        Runs all steps in sequence, yielding events for:
        - Pipeline start/end
        - Step start/end
        - Step-specific events (cache, policy, tools, etc.)
        - LLM streaming events (text deltas, tool calls)
        
        Events are collected in output.events and yielded after each step,
        as well as during streaming steps that emit events.
        
        Args:
            context: The agent run output (single source of truth) with is_streaming=True
            start_step_index: Index to start execution from (0-based). Used for HITL resumption.
            
        Yields:
            AgentEvent: Various event types providing visibility into execution
            
        Raises:
            Exception: Any exception from step execution is raised with proper error message
        """
        from upsonic.run.events.events import PipelineStartEvent, PipelineEndEvent, RunStartedEvent, RunCompletedEvent, StepStartEvent, StepEndEvent
        
        if not context.is_streaming:
            await self.execute(context, start_step_index)
            return
        
        if start_step_index > 0:
            if self.debug:
                from upsonic.utils.printing import info_log
                step_name = self.steps[start_step_index].name if start_step_index < len(self.steps) else "unknown"
                info_log(f"Resuming streaming pipeline from step {start_step_index} ({step_name})", "PipelineManager")
        
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        if not context.execution_stats:
            context.execution_stats = PipelineExecutionStats(total_steps=len(self.steps))
        
        pipeline_start_time = time.time()
        
        task_description = None
        if self.task:
            task_description = str(getattr(self.task, 'description', ''))[:200]
        
        run_id = None
        if self.agent and hasattr(self.agent, 'run_id'):
            run_id = self.agent.run_id
        
        agent_id = None
        if self.agent and hasattr(self.agent, 'agent_id'):
            agent_id = self.agent.agent_id
        
        # Emit RunStartedEvent FIRST - before pipeline
        yield RunStartedEvent(
            run_id=run_id or "",
            agent_id=agent_id or "",
            task_description=task_description
        )
        
        # Emit pipeline start event
        yield PipelineStartEvent(
            run_id=run_id or "",
            total_steps=len(self.steps),
            is_streaming=True,
            task_description=task_description
        )
        
        if self.debug:
            from upsonic.utils.printing import pipeline_started
            pipeline_started(len(self.steps))
        
        error_message = None
        
        with self._otel.pipeline_span(
            total_steps=len(self.steps), is_streaming=True, debug=self.debug,
        ) as otel_pipeline_span:
          try:
            for step_index in range(start_step_index, len(self.steps)):
                step = self.steps[step_index]
                
                if self.debug:
                    from upsonic.utils.printing import pipeline_step_started
                    pipeline_step_started(step.name, step.description)
                
                step_start_time = time.time()
                
                events_before = len(context.events)
                
                with self._otel.step_span(step.name, step.description) as otel_step_span:
                    if step.supports_streaming and context.is_streaming:
                        async for event in step.run_stream(context, self.task, self.agent, self.model, step_index, pipeline_manager=self):
                            yield event
                            if event not in context.events:
                                context.events.append(event)
                    else:
                        yield StepStartEvent(
                            run_id=run_id or "",
                            step_name=step.name,
                            step_index=step_index,
                            total_steps=len(self.steps),
                            step_description=step.description
                        )
                        
                        result = await step.run(context, self.task, self.agent, self.model, step_index, pipeline_manager=self)
                        
                        for event in context.events[events_before:]:
                            yield event
                        
                        step_result = context.step_results[-1] if context.step_results else None
                        yield StepEndEvent(
                            run_id=run_id or "",
                            step_name=step.name,
                            step_index=step_index,
                            status=step_result.status.value if step_result else "unknown",
                            execution_time=step_result.execution_time if step_result else time.time() - step_start_time,
                            message=step_result.message if step_result else None
                        )
                    
                    step_result_for_otel = context.step_results[-1] if context.step_results else None
                    if step_result_for_otel:
                        self._otel.set_step_result(
                            otel_step_span,
                            status=step_result_for_otel.status.value,
                            execution_time=step_result_for_otel.execution_time,
                            error_message=step_result_for_otel.message if step_result_for_otel.status == StepStatus.ERROR else None,
                        )
                
                result = context.step_results[-1] if context.step_results else None
                
                if self.debug and result:
                    from upsonic.utils.printing import pipeline_step_completed
                    pipeline_step_completed(
                        step.name, 
                        result.status.value, 
                        result.execution_time, 
                        result.message
                    )

            total_time = time.time() - pipeline_start_time
            
            last_step_status = context.step_results[-1].status if context.step_results else StepStatus.COMPLETED
            
            if last_step_status == StepStatus.COMPLETED:
                if self.agent and self.agent._agent_run_output:
                    self.agent._agent_run_output.mark_completed()
                    context.tool_call_count = getattr(self.agent, '_tool_call_count', 0)
                    context.tool_limit_reached = getattr(self.agent, '_tool_limit_reached', False)
            
            if self.debug:
                from upsonic.utils.printing import pipeline_completed, pipeline_timeline
                pipeline_completed(
                    context.execution_stats.executed_steps if context.execution_stats else 0,
                    len(self.steps),
                    total_time
                )
                step_results_dict = {sr.name: {"status": sr.status.value, "message": sr.message, "execution_time": sr.execution_time} for sr in context.step_results}
                pipeline_timeline(step_results_dict, total_time)
            
            self._otel.mark_success(otel_pipeline_span)
            
          except Exception as e:
            from upsonic.exceptions import RunCancelledException
            from upsonic.run.events.events import RunCancelledEvent
            
            self._otel.record_error(otel_pipeline_span, e)
            
            if isinstance(e, RunCancelledException):
                cancelled_step = context.get_cancelled_step()
                step_name = cancelled_step.name if cancelled_step else None
                yield RunCancelledEvent(
                    run_id=run_id or "",
                    message=str(e),
                    step_name=step_name
                )
            
            error_message = str(e)
            
            if self.agent and self.agent._agent_run_output:
                if isinstance(e, RunCancelledException):
                    self.agent._agent_run_output.mark_cancelled()
                else:
                    self.agent._agent_run_output.mark_error()
                
                context.tool_call_count = getattr(self.agent, '_tool_call_count', 0)
                context.tool_limit_reached = getattr(self.agent, '_tool_limit_reached', False)
            
            if self.debug:
                from upsonic.utils.printing import pipeline_failed
                pipeline_failed(
                    str(e),
                    context.execution_stats.executed_steps if context.execution_stats else 0,
                    context.execution_stats.total_steps if context.execution_stats else len(self.steps),
                    None,
                    None
                )
            
            raise
        
          finally:
            total_time = time.time() - pipeline_start_time
            executed_steps = context.execution_stats.executed_steps if context.execution_stats else 0

            try:
                yield PipelineEndEvent(
                    run_id=run_id or "",
                    status=context.step_results[-1].status.value if context.step_results else "unknown",
                    total_duration=total_time,
                    total_steps=len(self.steps),
                    executed_steps=executed_steps,
                    error_message=error_message
                )

                if context.step_results and context.step_results[-1].status == StepStatus.COMPLETED:
                    if self.agent and self.agent._agent_run_output:
                        self.agent._agent_run_output.mark_completed()
                        context.tool_call_count = getattr(self.agent, '_tool_call_count', 0)
                        context.tool_limit_reached = getattr(self.agent, '_tool_limit_reached', False)
                        yield RunCompletedEvent(
                            run_id=run_id or "",
                            agent_id=agent_id or "",
                            output_preview=str(context.output)[:100] if context.output else None
                        )
            except GeneratorExit:
                pass
    
    def get_execution_stats(self, context: Optional["AgentRunOutput"] = None) -> Dict[str, Any]:
        """Get statistics about the last execution from context."""
        if context and context.execution_stats:
            step_results_dict = {
                sr.name: {
                    "status": sr.status.value,
                    "message": sr.message,
                    "execution_time": sr.execution_time
                }
                for sr in context.step_results
            }
            return {
                "total_steps": context.execution_stats.total_steps,
                "executed_steps": context.execution_stats.executed_steps,
                "step_results": step_results_dict,
            }
        return {
            "total_steps": len(self.steps),
            "executed_steps": 0,
            "step_results": {}
        }
    
    def find_step_index_by_name(self, step_name: str) -> int:
        """Find step index by step name."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                return i
        raise ValueError(f"Step '{step_name}' not found in pipeline")

    async def _handle_external_tool_pause(
        self, 
        output: "AgentRunOutput", 
        e: "ExternalExecutionPause", 
    ) -> None:
        """Handle ExternalExecutionPause exception (non-streaming).
        
        Creates RunRequirement for each external tool call. All context for resumption
        is already in AgentRunOutput (chat_history, step_results, etc.).
        """
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution
        
        if self.task:
            self.task.is_paused = True
            if hasattr(self.task, '_usage') and self.task._usage is not None:
                self.task._usage.stop_timer()
        
        paused_calls = e.paused_calls or []
        
        if not paused_calls:
            raise RuntimeError("ExternalExecutionPause must have paused_calls attached by ToolManager")

        requires_guardrail_authorization = getattr(self.agent, "guardrail_provider", None) is not None
        for paused_call in paused_calls:
            tool_execution = ToolExecution(
                tool_call_id=paused_call.tool_call_id,
                tool_name=paused_call.tool_name,
                tool_args=paused_call.tool_args,
                result=paused_call.result,
                external_execution_required=True,
                requires_guardrail_authorization=requires_guardrail_authorization,
            )
            
            requirement = RunRequirement(tool_execution=tool_execution)
            output.add_requirement(requirement)
        
        output.mark_paused("external_tool")
        
        await self._save_session(output)
        
        if self.debug:
            from upsonic.utils.printing import info_log
            info_log(f"External tool pause: {len(paused_calls)} tool(s) waiting", "PipelineManager")

    async def _handle_confirmation_pause(
        self,
        output: "AgentRunOutput",
        e: "ConfirmationPause",
    ) -> None:
        """Handle ConfirmationPause exception.
        
        Creates RunRequirement with requires_confirmation for each tool call
        that needs user approval.
        """
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution

        if self.task:
            self.task.is_paused = True
            if hasattr(self.task, '_usage') and self.task._usage is not None:
                self.task._usage.stop_timer()

        paused_calls = e.paused_calls or []
        if not paused_calls:
            raise RuntimeError("ConfirmationPause must have paused_calls attached by ToolManager")

        requires_guardrail_authorization = getattr(self.agent, "guardrail_provider", None) is not None
        for paused_call in paused_calls:
            tool_execution = ToolExecution(
                tool_call_id=paused_call.tool_call_id,
                tool_name=paused_call.tool_name,
                tool_args=paused_call.tool_args,
                requires_confirmation=True,
                requires_guardrail_authorization=requires_guardrail_authorization,
            )
            requirement = RunRequirement(tool_execution=tool_execution)
            output.add_requirement(requirement)

        output.mark_paused("confirmation")
        await self._save_session(output)

        if self.debug:
            from upsonic.utils.printing import info_log
            info_log(f"Confirmation pause: {len(paused_calls)} tool(s) awaiting approval", "PipelineManager")

    async def _handle_user_input_pause(
        self,
        output: "AgentRunOutput",
        e: "UserInputPause",
    ) -> None:
        """Handle UserInputPause exception.
        
        Creates RunRequirement with requires_user_input and user_input_schema
        for each tool call that needs user-provided values.
        """
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution

        if self.task:
            self.task.is_paused = True
            if hasattr(self.task, '_usage') and self.task._usage is not None:
                self.task._usage.stop_timer()

        paused_calls = e.paused_calls or []
        if not paused_calls:
            raise RuntimeError("UserInputPause must have paused_calls attached by ToolManager")

        requires_guardrail_authorization = getattr(self.agent, "guardrail_provider", None) is not None
        for paused_call in paused_calls:
            tool_execution = ToolExecution(
                tool_call_id=paused_call.tool_call_id,
                tool_name=paused_call.tool_name,
                tool_args=paused_call.tool_args,
                requires_user_input=True,
                user_input_schema=paused_call.user_input_schema,
                requires_guardrail_authorization=requires_guardrail_authorization,
            )
            requirement = RunRequirement(tool_execution=tool_execution)
            output.add_requirement(requirement)

        output.mark_paused("user_input")
        await self._save_session(output)

        if self.debug:
            from upsonic.utils.printing import info_log
            info_log(f"User input pause: {len(paused_calls)} tool(s) awaiting input", "PipelineManager")

    def set_manager(self, name: str, manager: Any) -> None:
        """
        Register a manager in the pipeline registry.
        
        Args:
            name: Name identifier for the manager (e.g., 'memory_manager')
            manager: The manager instance to register
        """
        self._managers[name] = manager
    
    def get_manager(self, name: str) -> Optional[Any]:
        """
        Retrieve a manager from the pipeline registry.
        
        Args:
            name: Name identifier for the manager (e.g., 'memory_manager')
            
        Returns:
            The manager instance if found, None otherwise
        """
        return self._managers.get(name)
    
    def has_manager(self, name: str) -> bool:
        """
        Check if a manager exists in the registry.
        
        Args:
            name: Name identifier for the manager
            
        Returns:
            True if manager exists, False otherwise
        """
        return name in self._managers
    
    def remove_manager(self, name: str) -> bool:
        """
        Remove a manager from the registry.
        
        Args:
            name: Name identifier for the manager
            
        Returns:
            True if manager was removed, False if not found
        """
        if name in self._managers:
            del self._managers[name]
            return True
        return False
    
    def clear_managers(self) -> None:
        """Clear all managers from the registry."""
        self._managers.clear()
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = [step.name for step in self.steps]
        return f"PipelineManager(steps={step_names})"
