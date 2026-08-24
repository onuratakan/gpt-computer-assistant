from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from upsonic.agent.agent import Agent
from upsonic.agent.deepagent.backends import (
    BackendProtocol,
    StateBackend,
)
from upsonic.agent.deepagent.tools import (
    FilesystemToolKit,
    PlanningToolKit,
    SubagentToolKit,
)
from upsonic.agent.deepagent.constants import (
    BASE_AGENT_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    DEFAULT_SUBAGENT_PROMPT,
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    WRITE_TODOS_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from upsonic.models import Model
    from upsonic.storage.memory.memory import Memory
    from upsonic.db.database import DatabaseBase
    from upsonic.tasks.tasks import Task
    from upsonic.graph.graph import State


class DeepAgent(Agent):
    """
    Deep Agent with advanced capabilities for complex multi-step tasks.
    
    Extends the base Agent class with:
    - **Planning System**: write_todos tool for task decomposition
    - **Filesystem Tools**: ls, read_file, write_file, edit_file, glob, grep
    - **Subagent Delegation**: task tool for delegating to pre-configured agents
    - **Persistent Memory**: Configurable storage backends
    
    Architecture:
    - Inherits all Agent capabilities
    - Adds filesystem and planning toolkits
    - Maintains list of subagent Agent instances
    - Uses standard pipeline (removed custom steps per requirements)
    
    Usage:
        ```python
        from upsonic.agent.deepagent import DeepAgent
        from upsonic import Agent, Task
        
        # Create DeepAgent
        agent = DeepAgent(model="openai/gpt-4o", subagents=[Agent(model="openai/gpt-4o-mini", name="researcher", role="Research Specialist", system_prompt="You are a research expert...")])
        
        # Optionally add custom subagents
        custom_agent = Agent(
            model="openai/gpt-4o-mini",
            name="researcher",
            role="Research Specialist",
            system_prompt="You are a research expert..."
        )
        agent.add_subagent(custom_agent)
        
        # Execute complex task
        result = agent.do(Task("Complex research task"))
        ```
    """
    
    def __init__(
        self,
        model: Union[str, "Model"] = "openai/gpt-4o",
        *,
        # All base Agent parameters
        name: Optional[str] = None,
        memory: Optional["Memory"] = None,
        db: Optional["DatabaseBase"] = None,
        debug: bool = False,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
        
        # DeepAgent-specific parameters
        filesystem_backend: Optional[BackendProtocol] = None,
        enable_planning: bool = True,
        enable_filesystem: bool = True,
        enable_subagents: Optional[bool] = None,  # Auto-set based on subagents parameter
        subagents: Optional[List[Agent]] = None,  # List of pre-configured Agent instances
        tool_call_limit: int = 20,
        
        # Pass through all other Agent parameters via **kwargs
        **kwargs
    ):
        """
        Initialize a DeepAgent with advanced capabilities.
        
        Args:
            model: Model identifier or Model instance
            name: Agent name
            memory: Memory instance
            db: Database instance
            debug: Enable debug logging
            tools: Additional user tools
            system_prompt: User's custom system prompt
            tool_call_limit: Maximum number of tool calls per run
            
            # DeepAgent Features
            filesystem_backend: Backend for filesystem storage (defaults to StateBackend)
            enable_planning: Enable write_todos planning tool (default: True)
            enable_filesystem: Enable filesystem tools (default: True)
            enable_subagents: Enable task tool (auto-set based on subagents parameter)
            subagents: List of pre-configured Agent instances to add as subagents

            
            **kwargs: All other Agent parameters
        """
        # Default to StateBackend
        if filesystem_backend is None:
            filesystem_backend = StateBackend()
        
        self.filesystem_backend = filesystem_backend
        
        # Auto-set enable_subagents based on whether subagents are provided
        if enable_subagents is None:
            # If user provides subagents, enable automatically
            # Otherwise, default to True (create general-purpose)
            if subagents is not None:
                enable_subagents = len(subagents) > 0
            else:
                enable_subagents = True  # Default: create general-purpose
        
        # Store DeepAgent configuration
        self.enable_planning = enable_planning
        self.enable_filesystem = enable_filesystem
        self.enable_subagents = enable_subagents
        
        # Initialize subagents list (MUST be List[Agent])
        self.subagents: List[Agent] = []
        
        # Add user-provided subagents if any
        if subagents:
            for subagent in subagents:
                if not isinstance(subagent, Agent):
                    raise TypeError(f"All subagents must be Agent instances, got {type(subagent)}")
                if not hasattr(subagent, 'name') or not subagent.name:
                    raise ValueError("All subagents must have a name attribute")
                self.subagents.append(subagent)
        
        # Build enhanced system prompt
        enhanced_system_prompt = self._build_deep_system_prompt(user_prompt=system_prompt)
        
        # Build deep agent tools
        deep_tools = []
        
        if enable_filesystem:
            self._filesystem_toolkit = FilesystemToolKit(filesystem_backend)
            deep_tools.append(self._filesystem_toolkit)
        else:
            self._filesystem_toolkit = None
        
        if enable_planning:
            self._planning_toolkit = PlanningToolKit()
            deep_tools.append(self._planning_toolkit)
        else:
            self._planning_toolkit = None
        
        # Combine user tools + deep tools
        all_tools = deep_tools + (list(tools) if tools else [])
        
        # Store debug_level before calling super() to pass it through
        self._debug_level = kwargs.pop('debug_level', 1) if debug else 1
        
        # Initialize base Agent
        super().__init__(
            model=model,
            name=name,
            memory=memory,
            db=db,
            debug=debug,
            debug_level=self._debug_level,
            tools=all_tools,
            system_prompt=enhanced_system_prompt,
            tool_call_limit=tool_call_limit,
            **kwargs
        )
        
        # Create default general-purpose subagent if subagents enabled
        # and no subagents were provided by user
        if enable_subagents:
            # Check if user already provided a general-purpose subagent
            has_general_purpose = any(s.name == "general-purpose" for s in self.subagents)
            
            if not has_general_purpose:
                # Create default general-purpose subagent
                # Filter out SubagentToolKit from parent tools
                general_purpose_tools = [
                    t for t in self.tools
                    if not isinstance(t, SubagentToolKit)
                ]
                
                self._general_purpose_agent = Agent(
                    model=self.model,
                    name="general-purpose",
                    role="General Purpose Assistant",
                    goal=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,  # Use constant!
                    system_prompt=DEFAULT_SUBAGENT_PROMPT,
                    tools=general_purpose_tools,
                    memory=None,  # Complete isolation
                    debug=self.debug,
                )
                
                # Add to subagents list
                self.subagents.append(self._general_purpose_agent)
            
            # NOW add SubagentToolKit (after subagents configured)
            self._subagent_toolkit = SubagentToolKit(parent_agent=self)
            self.add_tools(self._subagent_toolkit)
        else:
            self._subagent_toolkit = None
    
    def _build_deep_system_prompt(self, user_prompt: Optional[str]) -> str:
        """
        Build enhanced system prompt with deep agent instructions.
        
        Args:
            user_prompt: User's custom system prompt
            
        Returns:
            Enhanced system prompt
        """
        parts = []
        
        # Base agent prompt
        parts.append(BASE_AGENT_PROMPT)
        
        # User's custom prompt
        if user_prompt:
            parts.append(user_prompt)
        
        # Filesystem instructions
        if self.enable_filesystem:
            parts.append(FILESYSTEM_SYSTEM_PROMPT)
        
        # Planning instructions (if enabled)
        if self.enable_planning:
            parts.append(WRITE_TODOS_SYSTEM_PROMPT)
        
        # Subagent instructions
        if self.enable_subagents:
            parts.append(TASK_SYSTEM_PROMPT)
        
        return "\n\n".join(parts)
    
    def add_subagent(self, agent: Agent) -> None:
        """
        Add a custom subagent to the DeepAgent.
        
        Args:
            agent: Agent instance to add as subagent
                  Must have .name attribute for identification
        
        Example:
            ```python
            deep_agent = DeepAgent(model="openai/gpt-4o")
            
            custom_agent = Agent(
                model="openai/gpt-4o-mini",
                name="researcher",
                role="Research Specialist",
                system_prompt="You are a research expert...",
                tools=[...]
            )
            
            deep_agent.add_subagent(custom_agent)
            ```
        """
        if not isinstance(agent, Agent):
            raise TypeError(f"Subagent must be an Agent instance, got {type(agent)}")
        
        if not hasattr(agent, 'name') or not agent.name:
            raise ValueError("Subagent must have a name for identification")
        
        # Check if name already exists
        existing_names = [s.name for s in self.subagents]
        if agent.name in existing_names:
            raise ValueError(f"Subagent with name '{agent.name}' already exists")
        
        # Add to subagents list
        self.subagents.append(agent)
        
        # Update task tool docstring
        if self._subagent_toolkit:
            self._subagent_toolkit._update_task_tool_docstring()
    
    def get_subagent_names(self) -> List[str]:
        """
        Get list of available subagent names.
        
        Returns:
            List of subagent names
        """
        return [s.name for s in self.subagents]
    
    def get_current_plan(self) -> List[Dict[str, Any]]:
        """
        Get the current todo list from the active task.
        
        Returns:
            List of todo dictionaries, or empty list if no plan
        """
        if not self.enable_planning or not self._planning_toolkit:
            return []
        
        return self._planning_toolkit.get_current_todos()
    
    def get_filesystem_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the filesystem backend.
        
        Returns:
            Dictionary with filesystem statistics
        """
        if hasattr(self.filesystem_backend, 'get_stats'):
            return self.filesystem_backend.get_stats()
        return {}
