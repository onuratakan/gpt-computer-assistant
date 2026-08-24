from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union, TYPE_CHECKING

from upsonic.agent.agent import Agent

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.storage.base import Storage
    from upsonic.storage.memory.memory import Memory
    from upsonic.canvas.canvas import Canvas
    from upsonic.models.settings import ModelSettings
    from upsonic.profiles import ModelProfile
    from upsonic.reflection import ReflectionConfig
    from upsonic.safety_engine.base import Policy
    from upsonic.models import Model
    from upsonic.culture.culture import Culture
    from upsonic.db.database import DatabaseBase
    from upsonic.skills import Skills
    from upsonic.models.instrumented import InstrumentationSettings
    from upsonic.integrations.tracing import TracingProvider
    from upsonic.integrations.promptlayer import PromptLayer
    from upsonic.guardrails import GuardrailProvider


RetryMode = Literal["raise", "return_false"]


class AutonomousAgent(Agent):
    """
    A pre-configured AI Agent with filesystem and shell capabilities.
    
    AutonomousAgent inherits from Agent and provides:
    - **Default Storage**: Uses InMemoryStorage automatically if no storage is provided
    - **Default Memory**: Creates Memory instance with session persistence
    - **Filesystem Tools**: Read, write, edit, search, list, move, copy, delete files
    - **Shell Tools**: Execute terminal commands with timeout and output capture
    - **Workspace Sandboxing**: All file/shell operations are restricted to workspace
    - **Printing**: ``print`` defaults to ``True``, so :meth:`do` / :meth:`do_async` emit the
      same Task/Agent metrics and stream panels as :meth:`print_do` unless you pass
      ``print=False``, ``print=None`` (match base :class:`Agent` behavior), or set
      ``UPSONIC_AGENT_PRINT=false``. Heartbeat runs stay non-printing.
    
    This is the ideal choice for:
    - Coding assistants that need to read/write files
    - DevOps automation agents
    - System administration tasks
    - Any task requiring filesystem or shell access
    
    Usage:
        ```python
        from upsonic import AutonomousAgent
        
        # Simple usage with defaults
        agent = AutonomousAgent(
            model="openai/gpt-4o",
            workspace="/path/to/project"
        )
        
        result = agent.do("Read the main.py file and add error handling")
        
        # Advanced usage with custom configuration
        agent = AutonomousAgent(
            model="anthropic/claude-sonnet-4-20250514",
            workspace="/path/to/project",
            name="CodeAssistant",
            enable_filesystem=True,
            enable_shell=True,
            shell_timeout=60,
            full_session_memory=True,  # Enable chat history
        )
        ```
    
    Attributes:
        autonomous_workspace: Path to the workspace directory (all operations sandboxed here)
        filesystem_toolkit: Filesystem operations toolkit (if enabled)
        shell_toolkit: Shell command execution toolkit (if enabled)
    """
    
    def __init__(
        self,
        model: Union[str, "Model"] = "openai/gpt-4o",
        *,
        workspace: Optional[str] = None,
        name: Optional[str] = None,
        # Storage/Memory configuration
        storage: Optional["Storage"] = None,
        memory: Optional["Memory"] = None,
        db: Optional["DatabaseBase"] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        # Memory features (save flags)
        full_session_memory: bool = True,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        # Memory features (load flags)
        load_full_session_memory: bool = True,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type["BaseModel"]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        feed_tool_call_results: Optional[bool] = None,
        # Toolkit configuration
        enable_filesystem: bool = True,
        enable_shell: bool = True,
        shell_timeout: int = 120,
        shell_max_output: int = 10000,
        blocked_commands: Optional[List[str]] = None,
        # Standard Agent parameters
        debug: bool = False,
        debug_level: int = 1,
        print: Optional[bool] = True,
        company_url: Optional[str] = None,
        company_objective: Optional[str] = None,
        company_description: Optional[str] = None,
        company_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reflection: bool = False,
        context_management: bool = False,
        context_management_keep_recent: int = 5,
        context_management_model: Optional[str] = None,
        reliability_layer: Optional[Any] = None,
        agent_id_: Optional[str] = None,
        canvas: Optional["Canvas"] = None,
        retry: int = 1,
        mode: RetryMode = "raise",
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[str] = None,
        education: Optional[str] = None,
        work_experience: Optional[str] = None,
        show_tool_calls: bool = True,
        tool_call_limit: int = 100,
        enable_thinking_tool: bool = False,
        enable_reasoning_tool: bool = False,
        tools: Optional[List[Any]] = None,
        skills: Optional["Skills"] = None,
        user_policy: Optional[Union["Policy", List["Policy"]]] = None,
        agent_policy: Optional[Union["Policy", List["Policy"]]] = None,
        tool_policy_pre: Optional[Union["Policy", List["Policy"]]] = None,
        tool_policy_post: Optional[Union["Policy", List["Policy"]]] = None,
        guardrail_provider: Optional["GuardrailProvider"] = None,
        user_policy_feedback: bool = False,
        agent_policy_feedback: bool = False,
        user_policy_feedback_loop: int = 1,
        agent_policy_feedback_loop: int = 1,
        settings: Optional["ModelSettings"] = None,
        profile: Optional["ModelProfile"] = None,
        reflection_config: Optional["ReflectionConfig"] = None,
        model_selection_criteria: Optional[Dict[str, Any]] = None,
        use_llm_for_selection: bool = False,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        reasoning_summary: Optional[Literal["concise", "detailed"]] = None,
        thinking_enabled: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        thinking_include_thoughts: Optional[bool] = None,
        reasoning_format: Optional[Literal["hidden", "raw", "parsed"]] = None,
        culture: Optional["Culture"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        instrument: Union[bool, "TracingProvider", "InstrumentationSettings", None] = None,
        promptlayer: Optional["PromptLayer"] = None,
        heartbeat: bool = False,
        heartbeat_period: int = 30,
        heartbeat_message: str = "",
    ) -> None:
        """
        Initialize AutonomousAgent with default storage, memory, and tools.
        
        Args:
            model: Model identifier or Model instance (default: "openai/gpt-4o")
            workspace: Workspace directory path. Defaults to current working directory.
                All file and shell operations are restricted to this directory.
            name: Agent name for identification
            
            # Storage/Memory configuration
            storage: Custom storage backend. If None, InMemoryStorage is created.
            memory: Custom Memory instance. If None, Memory is created with storage.
            db: Database instance (overrides memory if provided)
            session_id: Session identifier. Auto-generated if None.
            user_id: User identifier. Auto-generated if None.
            
            # Memory features
            full_session_memory: Enable chat history persistence (default: True)
            summary_memory: Enable session summary generation (default: False)
            user_analysis_memory: Enable user profile extraction (default: False)
            user_profile_schema: Pydantic model for user profile structure
            dynamic_user_profile: Generate profile schema dynamically
            num_last_messages: Limit on message turns to keep in history
            feed_tool_call_results: Include tool call results in history
            
            # Toolkit configuration
            enable_filesystem: Enable filesystem tools (default: True)
            enable_shell: Enable shell command tools (default: True)
            shell_timeout: Default timeout for shell commands in seconds (default: 120)
            shell_max_output: Maximum output length before truncation (default: 10000)
            blocked_commands: List of command patterns to block for security
            
            # Heartbeat configuration
            heartbeat: Enable periodic heartbeat execution (default: False).
                When True, interfaces will periodically send heartbeat_message to
                this agent and forward the response through the interface channel.
            heartbeat_period: Heartbeat interval in minutes (default: 30).
            heartbeat_message: Message to send to the agent on each heartbeat tick.
            
            print: Defaults to ``True`` so :meth:`do` / :meth:`do_async` print like
                :meth:`print_do`. Pass ``False`` for quiet runs, or ``None`` for the same
                resolution as :class:`Agent` (quiet ``do`` unless ``UPSONIC_AGENT_PRINT``).
            
            # Standard Agent parameters - see Agent class for full documentation
        """
        from upsonic.storage import InMemoryStorage, Memory
        from .filesystem_toolkit import AutonomousFilesystemToolKit
        from .shell_toolkit import AutonomousShellToolKit
        
        if workspace is not None:
            self.autonomous_workspace: Path = Path(workspace).resolve()
        else:
            self.autonomous_workspace = Path.cwd().resolve()
        
        if not self.autonomous_workspace.exists():
            self.autonomous_workspace.mkdir(parents=True, exist_ok=True)
        
        effective_storage: Optional[Storage]
        if db is not None:
            effective_storage = None
        elif memory is not None:
            effective_storage = memory.storage
        elif storage is not None:
            effective_storage = storage
        else:
            effective_storage = InMemoryStorage()
        
        self._autonomous_storage: Optional[Storage] = effective_storage
        
        effective_memory: Optional[Memory]
        if db is not None:
            effective_memory = None
        elif memory is not None:
            effective_memory = memory
        elif effective_storage is not None:
            effective_memory = Memory(
                storage=effective_storage,
                session_id=session_id,
                user_id=user_id,
                full_session_memory=full_session_memory,
                summary_memory=summary_memory,
                user_analysis_memory=user_analysis_memory,
                load_full_session_memory=load_full_session_memory,
                load_summary_memory=load_summary_memory,
                load_user_analysis_memory=load_user_analysis_memory,
                user_profile_schema=user_profile_schema,
                dynamic_user_profile=dynamic_user_profile,
                num_last_messages=num_last_messages,
                model=model if isinstance(model, str) else None,
                debug=debug,
                debug_level=debug_level,
                feed_tool_call_results=feed_tool_call_results if feed_tool_call_results is not None else False,
            )
        else:
            effective_memory = None
        
        # We don't cache `effective_memory`; the base class owns `self.memory`
        # and the `autonomous_memory` property routes through it (no stale snapshot).

        self.filesystem_toolkit: Optional[AutonomousFilesystemToolKit] = None
        self.shell_toolkit: Optional[AutonomousShellToolKit] = None
        
        default_tools: List[Any] = []
        
        if enable_filesystem:
            self.filesystem_toolkit = AutonomousFilesystemToolKit(workspace=self.autonomous_workspace)
            default_tools.append(self.filesystem_toolkit)
        
        if enable_shell:
            self.shell_toolkit = AutonomousShellToolKit(
                workspace=self.autonomous_workspace,
                default_timeout=shell_timeout,
                max_output_length=shell_max_output,
                blocked_commands=blocked_commands,
            )
            default_tools.append(self.shell_toolkit)
        
        all_tools = default_tools + (tools or [])
        
        effective_system_prompt = self._build_autonomous_system_prompt(
            user_system_prompt=system_prompt,
            enable_filesystem=enable_filesystem,
            enable_shell=enable_shell,
        )
        
        self.heartbeat: bool = heartbeat
        self.heartbeat_period: int = heartbeat_period
        self.heartbeat_message: str = heartbeat_message
        
        super().__init__(
            model=model,
            name=name,
            memory=effective_memory,
            db=db,
            session_id=session_id,
            user_id=user_id,
            debug=debug,
            debug_level=debug_level,
            print=print,
            company_url=company_url,
            company_objective=company_objective,
            company_description=company_description,
            company_name=company_name,
            system_prompt=effective_system_prompt,
            reflection=reflection,
            context_management=context_management,
            context_management_keep_recent=context_management_keep_recent,
            context_management_model=context_management_model,
            reliability_layer=reliability_layer,
            agent_id_=agent_id_,
            canvas=canvas,
            retry=retry,
            mode=mode,
            role=role,
            goal=goal,
            instructions=instructions,
            education=education,
            work_experience=work_experience,
            feed_tool_call_results=feed_tool_call_results,
            show_tool_calls=show_tool_calls,
            tool_call_limit=tool_call_limit,
            enable_thinking_tool=enable_thinking_tool,
            enable_reasoning_tool=enable_reasoning_tool,
            tools=all_tools,
            skills=skills,
            user_policy=user_policy,
            agent_policy=agent_policy,
            tool_policy_pre=tool_policy_pre,
            tool_policy_post=tool_policy_post,
            guardrail_provider=guardrail_provider,
            user_policy_feedback=user_policy_feedback,
            agent_policy_feedback=agent_policy_feedback,
            user_policy_feedback_loop=user_policy_feedback_loop,
            agent_policy_feedback_loop=agent_policy_feedback_loop,
            settings=settings,
            profile=profile,
            reflection_config=reflection_config,
            model_selection_criteria=model_selection_criteria,
            use_llm_for_selection=use_llm_for_selection,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            thinking_include_thoughts=thinking_include_thoughts,
            reasoning_format=reasoning_format,
            culture=culture,
            metadata=metadata,
            workspace=str(self.autonomous_workspace),
            instrument=instrument,
            promptlayer=promptlayer,
        )
    
    def _build_autonomous_system_prompt(
        self,
        user_system_prompt: Optional[str],
        enable_filesystem: bool,
        enable_shell: bool,
    ) -> Optional[str]:
        """
        Build the system prompt for AutonomousAgent.
        
        If user provides a custom system_prompt, it's used as-is.
        Otherwise, builds a dynamic prompt based on enabled toolkits.
        
        Args:
            user_system_prompt: User-provided system prompt (if any)
            enable_filesystem: Whether filesystem tools are enabled
            enable_shell: Whether shell tools are enabled
            
        Returns:
            The effective system prompt to use
        """
        # If user provides custom prompt, wrap and return
        if user_system_prompt is not None:
            return f"<AutonomousAgent>\n{user_system_prompt}\n</AutonomousAgent>"

        # If no tools enabled, no special prompt needed
        if not enable_filesystem and not enable_shell:
            return None
        
        # Build dynamic prompt based on enabled tools
        prompt_parts = [
            "You are an Autonomous Agent with access to tools for completing tasks within your designated workspace."
        ]
        
        # Capabilities section
        capabilities = []
        
        if enable_filesystem:
            capabilities.append("""### Filesystem Tools
You have comprehensive filesystem access within the workspace:

- **read_file**: Read file contents. Always read a file before editing it. Use offset/limit for large files.
- **write_file**: Create new files or completely overwrite existing files. Parent directories are created automatically.
- **edit_file**: Make precise text replacements in files. You MUST read the file first before editing.
- **list_files**: List directory contents. Use pattern parameter for filtering (e.g., "*.py").
- **search_files**: Find files by name pattern across the workspace.
- **grep_files**: Search for text/regex patterns within file contents.
- **file_info**: Get file metadata (size, permissions, modification time).
- **create_directory**: Create directories (parents created automatically).
- **move_file**: Move or rename files and directories.
- **copy_file**: Copy files or directories.
- **delete_file**: Delete files or directories.""")
        
        if enable_shell:
            capabilities.append("""### Shell Tools
You can execute commands in the workspace directory:

- **run_command**: Execute shell commands (ls, git, pip, npm, etc.). Commands run in the workspace directory.
- **run_python**: Execute Python code snippets and get the output.
- **check_command_exists**: Verify if a command is available on the system.""")
        
        if capabilities:
            prompt_parts.append("\n## Your Capabilities\n")
            prompt_parts.append("\n\n".join(capabilities))
        
        # Guidelines section
        guidelines = []
        
        if enable_filesystem:
            guidelines.append("""### When to Use Filesystem Tools
- **Reading code**: Use read_file to understand existing code before making changes.
- **Creating files**: Use write_file for new files. The file path is relative to the workspace.
- **Editing code**: ALWAYS read_file first, then use edit_file with precise old_string/new_string.
- **Finding files**: Use search_files to locate files by name, grep_files to search content.
- **Project exploration**: Use list_files to understand project structure.""")
        
        if enable_shell:
            guidelines.append("""### When to Use Shell Tools
- **Running tests**: Use run_command to execute test suites (pytest, npm test, etc.).
- **Installing dependencies**: Use run_command for pip install, npm install, etc.
- **Git operations**: Use run_command for git status, git diff, git commit, etc.
- **Building/compiling**: Use run_command for build commands.
- **Quick Python snippets**: Use run_python for calculations or quick scripts.""")
        
        if guidelines:
            prompt_parts.append("\n## Guidelines\n")
            prompt_parts.append("\n\n".join(guidelines))
        
        # Best practices
        best_practices = ["### Best Practices"]
        practice_num = 1
        
        if enable_filesystem:
            best_practices.append(f"{practice_num}. **Read before edit**: Always read a file before attempting to edit it. This ensures you understand the current state.")
            practice_num += 1
            best_practices.append(f"{practice_num}. **Use precise edits**: When editing, provide enough context in old_string to uniquely identify the location.")
            practice_num += 1
        
        if enable_shell:
            best_practices.append(f"{practice_num}. **Check command availability**: Use check_command_exists before running commands that might not be installed.")
            practice_num += 1
        
        best_practices.append(f"{practice_num}. **Handle errors gracefully**: If a tool fails, analyze the error and try an alternative approach.")
        practice_num += 1
        best_practices.append(f"{practice_num}. **Work incrementally**: For complex tasks, break them into steps and verify each step works.")
        practice_num += 1
        
        if enable_filesystem:
            best_practices.append(f"{practice_num}. **Stay in workspace**: All paths are relative to the workspace. You cannot access files outside it.")
        
        prompt_parts.append("\n".join(best_practices))
        
        # Security section
        security_notes = ["\n## Security Restrictions"]
        if enable_filesystem:
            security_notes.append("- All file operations are sandboxed to the workspace directory.")
            security_notes.append("- Path traversal (../) outside the workspace is blocked.")
        if enable_shell:
            security_notes.append("- Dangerous shell commands are blocked for security.")
            security_notes.append("- Shell commands execute in the workspace directory.")
        
        prompt_parts.append("\n".join(security_notes))
        
        # Closing
        prompt_parts.append("\nWhen given a task, think about what tools you need and use them methodically to accomplish the goal.")

        body: str = "\n".join(prompt_parts)
        return f"<AutonomousAgent>\n{body}\n</AutonomousAgent>"
    
    @property
    def autonomous_storage(self) -> Optional["Storage"]:
        """The storage backend currently in use by this agent."""
        if self.memory is not None:
            return getattr(self.memory, "storage", None) or self._autonomous_storage
        return self._autonomous_storage

    @property
    def autonomous_memory(self) -> Optional["Memory"]:
        """The memory instance currently in use by this agent."""
        return self.memory
    
    def reset_filesystem_tracking(self) -> None:
        """
        Reset filesystem read tracking.
        
        This clears the record of which files have been read,
        which affects edit_file's read-before-edit enforcement.
        """
        if self.filesystem_toolkit:
            self.filesystem_toolkit.reset_read_tracking()
    
    async def aexecute_heartbeat(self) -> Optional[str]:
        """
        Execute the heartbeat message as a task and return the agent's response.
        
        Sends ``self.heartbeat_message`` to this agent via ``do_async`` and
        extracts the textual response.  Returns ``None`` when heartbeat is
        disabled, the message is empty, or the agent produces no output.
        
        Returns:
            The agent's text response, or None.
        """
        if not self.heartbeat or not self.heartbeat_message:
            return None

        from upsonic.tasks.tasks import Task

        task: Task = Task(self.heartbeat_message)
        saved_print_param: Optional[bool] = self._print_param
        saved_print_attr: bool = self.print
        try:
            self._print_param = False
            self.print = False
            await self.do_async(task, _print_method_default=False)
        finally:
            self._print_param = saved_print_param
            self.print = saved_print_attr

        run_result = self.get_run_output()
        if not run_result:
            return None

        model_response = run_result.get_last_model_response()
        if model_response and hasattr(model_response, "text") and model_response.text:
            return str(model_response.text)
        if run_result.output:
            return str(run_result.output)
        return None

    def execute_heartbeat(self) -> Optional[str]:
        """
        Execute the heartbeat message as a task and return the agent's response (sync).
        
        Synchronous wrapper around :meth:`aexecute_heartbeat`.
        
        Returns:
            The agent's text response, or None.
        """
        if not self.heartbeat or not self.heartbeat_message:
            return None

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.aexecute_heartbeat())
                return future.result()
        except RuntimeError:
            return asyncio.run(self.aexecute_heartbeat())

    def __repr__(self) -> str:
        """String representation of AutonomousAgent."""
        return (
            f"AutonomousAgent("
            f"model={self.model_name!r}, "
            f"workspace={str(self.autonomous_workspace)!r}, "
            f"name={self.name!r}, "
            f"tools={len(self.tools)}"
            f")"
        )
