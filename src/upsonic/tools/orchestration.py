"""Orchestration and planning tools for complex agent tasks."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from upsonic.tools.base import Tool
from upsonic.tools.config import tool

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.tools.registry import ToolRegistry


class _AgentReference:
    """Weak live-agent lookup with a serializable fallback agent."""

    def __init__(self, agent: Any, fallback_agent: Any = None) -> None:
        self._fallback_agent = fallback_agent if fallback_agent is not None else agent
        self._agent_ref = None
        self.bind(agent)

    def bind(self, agent: Any) -> None:
        import weakref

        try:
            self._agent_ref = weakref.ref(agent)
        except TypeError:
            self._agent_ref = None

    def __call__(self) -> Any:
        agent = self._agent_ref() if self._agent_ref is not None else None
        return agent if agent is not None else self._fallback_agent

    def __getstate__(self) -> Dict[str, Any]:
        return {"_fallback_agent": self._fallback_agent}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._agent_ref = None
        self._fallback_agent = state.get("_fallback_agent")


class PlanStep(BaseModel):
    """Single tool call in a high-level plan."""
    tool_name: str = Field(
        ..., 
        description="The exact name of the tool to be called for this step."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The dictionary of parameters to pass to the tool."
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of what this step accomplishes."
    )


class AnalysisResult(BaseModel):
    """Structured output of an automated analysis step."""
    evaluation: str = Field(
        ...,
        description="Detailed reasoning and evaluation of the last tool's result."
    )
    next_action: Literal['continue_plan', 'revise_plan', 'final_answer'] = Field(
        ...,
        description="Agent directive for the orchestrator: continue, revise, or finalize."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Additional reasoning for the chosen next action."
    )


class Thought(BaseModel):
    """Initial structured thinking process for the AI agent."""
    reasoning: str = Field(
        ...,
        description="Detailed explanation of understanding and strategy."
    )
    plan: List[PlanStep] = Field(
        ...,
        description="Step-by-step execution plan of tool calls."
    )
    criticism: str = Field(
        ...,
        description="Self-critique identifying potential flaws or ambiguities."
    )
    action: Literal['execute_plan', 'request_clarification'] = Field(
        'execute_plan',
        description="Next action: execute the plan or request clarification."
    )
    clarification_needed: Optional[str] = Field(
        None,
        description="Specific clarification needed if action is 'request_clarification'."
    )


class ExecutionResult(BaseModel):
    """Result of executing an orchestrated plan."""
    
    success: bool = Field(
        ...,
        description="Whether the plan execution was successful."
    )
    final_result: Any = Field(
        ...,
        description="The final synthesized result."
    )
    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of all tool executions."
    )
    total_steps: int = Field(
        ...,
        description="Total number of steps executed."
    )
    revisions: int = Field(
        0,
        description="Number of plan revisions made."
    )


@tool(
    requires_confirmation=False,
    show_result=False,
    sequential=True,
    docstring_format='google'
)
def plan_and_execute(thought: Thought) -> str:
    """Master tool for complex tasks. Executes multi-step plans sequentially.
    
    Args:
        thought: Structured thought object with reasoning, plan, and criticism.
    
    Returns:
        Placeholder string - actual execution handled by orchestrator.
    """
    # This is a pseudo-tool - actual implementation is in the processor
    return "Plan received and will be executed by the orchestrator."


class Orchestrator(Tool):
    """Orchestrator for complex multi-step tool executions with optional reasoning."""
    
    def __init__(
        self,
        agent_instance: Any,
        task: Optional['Task'],
        wrapped_tools: Dict[str, Callable],
        live_agent_instance: Any = None,
    ):
        """Initialize the orchestrator."""
        # Initialize Tool base class
        super().__init__(
            name="orchestrator",
            description="Orchestrates multi-step tool execution with reasoning",
            tool_id=f"Orchestrator_{id(agent_instance)}"  # Unique per agent instance
        )
        
        self.agent_instance = agent_instance
        self._guardrail_agent_ref = _AgentReference(
            live_agent_instance or agent_instance,
            fallback_agent=agent_instance,
        )
        self.task = task
        self.wrapped_tools = wrapped_tools
        self.is_reasoning_enabled = agent_instance.enable_reasoning_tool if agent_instance else False
        self.original_user_request = task.description if task else ""
        
        self.execution_history = f"Orchestrator's execution history for the user's request:\n"
        self.program_counter = 0
        self.pending_plan = []
        self.revision_count = 0
        
        self.all_tools = {
            name: func 
            for name, func in wrapped_tools.items()
            if name != 'plan_and_execute'
        }

    def bind_guardrail_provider_parent(self, agent: Any) -> None:
        self._guardrail_agent_ref.bind(agent)

    def _guardrail_agent(self) -> Any:
        return self._guardrail_agent_ref()
    
    async def execute(self, thought: Thought) -> Any:
        """Main entry point for orchestrator execution."""
        from upsonic.utils.printing import console, spacing
        
        console.print("[bold magenta]Orchestrator Activated:[/bold magenta] Received initial plan.")
        spacing()
        
        if not self.agent_instance:
            return "Error: Orchestrator was not properly initialized with an agent instance."
        
        self.execution_history += f"Initial Thought & Plan: {thought.plan}\nReasoning: {thought.reasoning}\n Criticism: {thought.criticism}\n\n"
        self.pending_plan = thought.plan
        self.program_counter = 0
        
        while self.program_counter < len(self.pending_plan):
            step = self.pending_plan[self.program_counter]
            
            result = await self._execute_single_step(step)
            
            if self.is_reasoning_enabled:
                should_continue = await self._handle_reasoning_step(step.tool_name, result)
                if not should_continue:
                    break
            else:
                self.program_counter += 1
        
        return await self._synthesize_final_answer()
    
    async def _execute_single_step(self, step: PlanStep) -> Any:
        """Execute a single tool step from the plan."""
        from upsonic.utils.printing import console, print_orchestrator_tool_step
        
        tool_name = step.tool_name.split('.')[-1]
        params = step.parameters
        
        console.print(
            f"[bold blue]Executing Tool Step {self.program_counter + 1}/{len(self.pending_plan)}:[/bold blue] "
            f"Calling tool [cyan]{tool_name}[/cyan] with params {params}"
        )
        
        if tool_name not in self.all_tools:
            result = f"Error: Tool '{tool_name}' is not an available tool."
            console.print(f"[bold red]{result}[/bold red]")
        else:
            try:
                guardrail_agent = self._guardrail_agent()
                agent_attrs = getattr(guardrail_agent, "__dict__", {})
                has_guardrail_provider = (
                    isinstance(agent_attrs, dict)
                    and "guardrail_provider" in agent_attrs
                ) or hasattr(type(guardrail_agent), "guardrail_provider")
                guardrail_provider = (
                    getattr(guardrail_agent, "guardrail_provider", None)
                    if has_guardrail_provider
                    else None
                )
                if guardrail_provider is not None and hasattr(guardrail_agent, "_execute_tool_calls"):
                    from upsonic.messages import ToolCallPart

                    tool_limit = getattr(guardrail_agent, "tool_call_limit", None)
                    attempt_count = getattr(guardrail_agent, "_tool_attempt_count", None)
                    if tool_limit and callable(attempt_count) and attempt_count() + 1 >= tool_limit:
                        guardrail_agent._tool_limit_reached = True
                        run_output = getattr(guardrail_agent, "_agent_run_output", None)
                        if run_output is not None:
                            run_output.tool_limit_reached = True
                        result = f"Tool call limit of {tool_limit} reached. Cannot execute more tools."
                    else:
                        tool_results = await guardrail_agent._execute_tool_calls([
                            ToolCallPart(tool_name=tool_name, args=params)
                        ])
                        result = tool_results[0].content if tool_results else None
                else:
                    tool_to_call = self.all_tools[tool_name]
                    result = await tool_to_call(**params)
            except Exception as e:
                error_message = f"An error occurred while executing tool '{tool_name}': {e}"
                console.print(f"[bold red]{error_message}[/bold red]")
                result = error_message
        
        print_orchestrator_tool_step(tool_name, params, result)
        self.execution_history += f"\nStep {self.program_counter + 1} (Tool: {tool_name}):\nResult: {result}\n"
        
        return result
    
    async def _inject_analysis(self) -> AnalysisResult:
        """Inject mandatory analysis step after tool execution."""
        from upsonic.tasks.tasks import Task
        from upsonic.agent.agent import Agent
        
        analysis_prompt = (
            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{self.original_user_request}'\n\n"
            "You are in the middle of a multi-step plan. An action has just been completed. You must now analyze the outcome before proceeding. "
            "Based on the execution history, evaluate the result of the last tool call and decide the "
            "most logical next action.\n\n"
            "CRITICAL: You are ONLY analyzing the results. DO NOT call any tools. DO NOT execute any actions. "
            "ONLY provide your evaluation based on the execution history below.\n\n"
            "<ExecutionHistory>\n"
            f"{self.execution_history}"
            "</ExecutionHistory>"
        )
        
        # Create analysis task with NO tools - analysis agent should only evaluate, not execute
        analysis_task = Task(
            description=analysis_prompt, 
            not_main_task=True, 
            response_format=AnalysisResult,
            tools=[]  # Explicitly no tools for analysis
        )
        
        # Create a fresh analysis agent with NO tools at all
        # We cannot use copy because it shares tool references
        analysis_agent = Agent(
            model=self.agent_instance.model,
            name=f"{self.agent_instance.name}_analysis",
            tools=[],  # NO tools for analysis
            enable_thinking_tool=False,
            enable_reasoning_tool=False
        )
        
        analysis_run_result = await analysis_agent.do_async(analysis_task, return_output=True)
        analysis_result: AnalysisResult = analysis_run_result.output if hasattr(analysis_run_result, 'output') else analysis_run_result
        self.execution_history += f"\n--- Injected Analysis ---\nEvaluation: {analysis_result.evaluation}\n"

        return analysis_result
    
    async def _handle_reasoning_step(self, tool_name: str, result: Any) -> bool:
        """Handle reasoning injection after tool execution."""
        from upsonic.utils.printing import console
        
        console.print(f"[bold yellow]Injecting Mandatory Analysis Step after Tool '{tool_name}'...[/bold yellow]")
        
        analysis_result = await self._inject_analysis()
        
        if analysis_result.next_action == 'continue_plan':
            console.print("[bold green]Analysis complete. Continuing with the original plan.[/bold green]")
            self.program_counter += 1
            return True
        
        elif analysis_result.next_action == 'final_answer':
            console.print("[bold green]Analysis concluded that the task is complete. Proceeding to final synthesis.[/bold green]")
            return False
        
        elif analysis_result.next_action == 'revise_plan':
            console.print("[bold red]Analysis concluded that the plan is flawed. Requesting a new plan.[/bold red]")
            await self._request_plan_revision()
            return True
        
        return True
    
    async def _request_plan_revision(self) -> None:
        """Request revised plan based on execution history."""
        from upsonic.tasks.tasks import Task
        from upsonic.utils.printing import console
        from upsonic.agent.agent import Agent
        
        revision_prompt = (
            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{self.original_user_request}'\n\n"
            "You are in the middle of a multi-step plan. Your own analysis has determined that the "
            "original plan is flawed or insufficient. Based on the *entire* execution history so far, "
            "formulate a new, complete `Thought` object with a better plan to achieve the user's "
            "original goal.\n\n"
            "CRITICAL: You are ONLY creating a new plan. DO NOT call any tools. DO NOT execute any actions. "
            "ONLY provide a revised Thought with a better plan based on the execution history.\n\n"
            "<ExecutionHistory>\n"
            f"{self.execution_history}"
            "</ExecutionHistory>"
        )
        
        # Create revision task with NO tools - revision agent should only plan, not execute
        revision_task = Task(
            description=revision_prompt, 
            not_main_task=True, 
            response_format=Thought,
            tools=[]  # Explicitly no tools for revision
        )
        
        # Create a fresh revision agent with NO tools at all
        revision_agent = Agent(
            model=self.agent_instance.model,
            name=f"{self.agent_instance.name}_revision",
            tools=[],  # NO tools for revision
            enable_thinking_tool=False,
            enable_reasoning_tool=False
        )
        
        revision_run_result = await revision_agent.do_async(revision_task, return_output=True)
        new_thought: Thought = revision_run_result.output if hasattr(revision_run_result, 'output') else revision_run_result

        console.print("[bold magenta]Orchestrator:[/bold magenta] Received revised plan. Restarting execution.")
        self.pending_plan = new_thought.plan
        self.program_counter = 0
        self.revision_count += 1
        self.execution_history += f"\n--- PLAN REVISED ---\nNew Reasoning: {new_thought.reasoning}\n"
    
    async def _synthesize_final_answer(self) -> Any:
        """Synthesize final answer based on execution history."""
        from upsonic.tasks.tasks import Task
        from upsonic.utils.printing import console, spacing
        from upsonic.agent.agent import Agent
        
        console.print("[bold magenta]Orchestrator:[/bold magenta] Plan complete. Preparing for final synthesis.")
        spacing()
        
        synthesis_prompt = (
            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{self.original_user_request}'\n\n"
            "You are in the final step of a multi-step task. "
            "You have already executed a plan and gathered all necessary information. "
            "Based *only* on the execution history provided below, synthesize a complete "
            "and final answer for the user's original request.\n\n"
            "CRITICAL: You are ONLY synthesizing the final answer. DO NOT call any tools. DO NOT execute any actions. "
            "ONLY provide a comprehensive summary based on the execution history.\n\n"
            "<ExecutionHistory>\n"
            f"{self.execution_history}"
            "</ExecutionHistory>"
        )
        
        # Create synthesis task with NO tools - synthesis agent should only summarize, not execute
        synthesis_task = Task(
            description=synthesis_prompt, 
            not_main_task=True,
            tools=[]  # Explicitly no tools for synthesis
        )
        
        # Create a fresh synthesis agent with NO tools at all
        synthesis_agent = Agent(
            model=self.agent_instance.model,
            name=f"{self.agent_instance.name}_synthesis",
            tools=[],  # NO tools for synthesis
            enable_thinking_tool=False,
            enable_reasoning_tool=False
        )
        
        final_response = await synthesis_agent.do_async(synthesis_task, return_output=True)

        return final_response.output if hasattr(final_response, 'output') else final_response


class OrchestratorLifecycle:
    """Manages the lifecycle of the ``plan_and_execute`` ``Orchestrator``.

    Owns ``Orchestrator`` create / update / discard around tool
    registration and removal. Reads ``wrapped_tools`` from a
    ``ToolRegistry`` reference so that the orchestrator can be wired up
    with a live reference to the manager's wrapped-tools dict.
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self._registry = registry
        self._orchestrator: Optional[Orchestrator] = None
        self._current_task: Optional["Task"] = None

    def get_orchestrator(self) -> Optional[Orchestrator]:
        return self._orchestrator

    def maybe_create(
        self,
        new_tools: Dict[str, Tool],
        task: Optional["Task"],
        agent_instance: Optional[Any],
        live_agent_instance: Optional[Any] = None,
    ) -> None:
        """Handle ``plan_and_execute`` orchestrator creation/update.

        If ``plan_and_execute`` was newly registered and the agent has
        ``enable_thinking_tool=True``, this creates or updates the
        ``Orchestrator`` and rewires ``wrapped_tools['plan_and_execute']``
        to the orchestrator's executor closure (overriding the regular
        behavioral wrapper installed earlier).
        """
        self._current_task = task

        wrapped_tools = self._registry.wrapped_tools

        if 'plan_and_execute' in new_tools:
            if agent_instance and agent_instance.enable_thinking_tool:
                if self._orchestrator:
                    if live_agent_instance is not None:
                        self._orchestrator.bind_guardrail_provider_parent(live_agent_instance)
                    self._orchestrator.wrapped_tools = wrapped_tools
                    self._orchestrator.all_tools = {
                        name: func
                        for name, func in wrapped_tools.items()
                        if name != 'plan_and_execute'
                    }
                    if task:
                        self._orchestrator.task = task
                        self._orchestrator.original_user_request = task.description
                else:
                    self._orchestrator = Orchestrator(
                        agent_instance=agent_instance,
                        task=task,
                        wrapped_tools=wrapped_tools,
                        live_agent_instance=live_agent_instance,
                    )

                orchestrator = self._orchestrator
                assert orchestrator is not None  # mypy narrowing

                async def orchestrator_executor(thought) -> Any:
                    return await orchestrator.execute(thought)

                wrapped_tools['plan_and_execute'] = orchestrator_executor
            # else: keep regular behavioral wrapper installed earlier

    def update_context(self, task: "Task") -> None:
        """Refresh the existing orchestrator with a new task context.

        Idempotent: if ``maybe_create`` already updated the orchestrator
        in this call, this re-applies the same values (no-op effect).
        """
        if not self._orchestrator or not task:
            return
        self._orchestrator.task = task
        self._orchestrator.original_user_request = task.description
        wrapped_tools = self._registry.wrapped_tools
        self._orchestrator.wrapped_tools = wrapped_tools
        self._orchestrator.all_tools = {
            name: func
            for name, func in wrapped_tools.items()
            if name != 'plan_and_execute'
        }

    def maybe_discard(self, removed_names: List[str]) -> None:
        """Discard or refresh the orchestrator after tools were removed.

        If ``plan_and_execute`` was removed, drop the orchestrator
        entirely. Otherwise refresh its tool maps to reflect the new
        registry state.
        """
        if not self._orchestrator:
            return

        if 'plan_and_execute' in removed_names:
            self._orchestrator = None
            return

        wrapped_tools = self._registry.wrapped_tools
        self._orchestrator.wrapped_tools = wrapped_tools
        self._orchestrator.all_tools = {
            name: func
            for name, func in wrapped_tools.items()
            if name != 'plan_and_execute'
        }

    
