"""
Subagent ToolKit - Delegate tasks to pre-configured subagent instances.

This toolkit provides the task tool which delegates to Agent instances
stored in the parent DeepAgent's subagents list.
"""

from typing import Any, List
from upsonic.tools import tool, ToolKit
from upsonic.agent.deepagent.constants import TASK_TOOL_DESCRIPTION


class SubagentToolKit(ToolKit):
    """
    Subagent delegation toolkit for DeepAgent.
    
    Provides the task tool which delegates to pre-configured Agent instances.
    Subagents are stored in parent.subagents (List[Agent]) and looked up by name.
    
    Architecture:
    - Subagents are real Agent instances (not configs)
    - Stored in DeepAgent.subagents list
    - task tool looks up by agent.name and executes
    - Tools already registered in each subagent's __init__
    """
    
    def __init__(self, parent_agent: Any, **kwargs: Any):
        """
        Initialize the subagent toolkit.
        
        Registers tools from subagents into the toolkit.
        
        Args:
            parent_agent: Reference to the parent DeepAgent instance
                         Must have .subagents attribute (List[Agent])
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)
        self.agent = parent_agent
        
        # Update task tool docstring with available subagents
        self._update_task_tool_docstring()
    
    def _update_task_tool_docstring(self) -> None:
        """
        Update the task tool's docstring using TASK_TOOL_DESCRIPTION constant.
        
        Fills in the {available_agents} placeholder with subagent descriptions.
        """
        # Build available agents list from subagents
        agents_list = []
        
        if hasattr(self.agent, 'subagents') and self.agent.subagents:
            for subagent in self.agent.subagents:
                agent_name = getattr(subagent, 'name', 'unnamed')
                
                # Build description from agent attributes
                description_parts = []
                
                if hasattr(subagent, 'role') and subagent.role:
                    description_parts.append(f"Role: {subagent.role}")
                
                if hasattr(subagent, 'goal') and subagent.goal:
                    description_parts.append(f"Goal: {subagent.goal}")
                
                if hasattr(subagent, 'system_prompt') and subagent.system_prompt:
                    description_parts.append(f"Specialty: {subagent.system_prompt}...")
                
                description = " | ".join(description_parts) if description_parts else "General purpose agent"
                
                agents_list.append(f'"{agent_name}": {description}')
        
        if not agents_list:
            agents_list.append('"general-purpose": No subagents configured')
        
        available_agents = "\n".join(agents_list)
        
        # Use TASK_TOOL_DESCRIPTION explicitly as required
        self.task.__func__.__doc__ = TASK_TOOL_DESCRIPTION.format(available_agents=available_agents)
    
    @tool(timeout=200)
    async def task(
        self,
        task_description: str,
        subagent_type: str = "general-purpose"
    ) -> str:
        """Placeholder docstring - will be replaced from constants."""
        # Validate subagents exist
        if not hasattr(self.agent, 'subagents') or not self.agent.subagents:
            return (
                "❌ Error: No subagents configured\n\n"
                "The task tool requires subagents to be configured in DeepAgent.\n"
                "Please configure subagents when creating DeepAgent."
            )
        
        # Find subagent by name
        selected_subagent = None
        for subagent in self.agent.subagents:
            if getattr(subagent, 'name', None) == subagent_type:
                selected_subagent = subagent
                break
        
        if selected_subagent is None:
            available = [getattr(s, 'name', 'unnamed') for s in self.agent.subagents]
            return (
                f"❌ Error: Subagent '{subagent_type}' not found\n\n"
                f"Available subagents: {', '.join(available)}\n\n"
                f"Please use one of the available subagent names."
            )
        
        try:
            # Create task for the subagent
            from upsonic import Task
            
            subtask = Task(
                description=task_description,
                not_main_task=True  # Don't print summaries
            )
            
            # Execute using the pre-configured subagent. If the parent has a
            # guardrail provider and the child does not, bind it to this
            # invocation without mutating a potentially shared child agent.
            selected_subagent = self._subagent_for_execution(selected_subagent)
            result = await selected_subagent.do_async(subtask)
            
            # Return simple output
            return str(result) if result is not None else "No response from subagent"
            
        except Exception as e:
            # Return error to LLM (don't raise)
            return f"❌ Subagent execution failed: {str(e)}"

    def _subagent_for_execution(self, subagent: Any) -> Any:
        provider = getattr(self.agent, "guardrail_provider", None)
        if provider is None or getattr(subagent, "guardrail_provider", None) is not None:
            return subagent

        from upsonic.tools.wrappers import AgentTool

        return AgentTool(subagent, guardrail_provider=provider)._agent_for_execution()
    
    def get_subagent_names(self) -> List[str]:
        """
        Get list of available subagent names.
        
        Returns:
            List of subagent names
        """
        if not hasattr(self.agent, 'subagents') or not self.agent.subagents:
            return []
        
        return [getattr(s, 'name', 'unnamed') for s in self.agent.subagents]


# Set the task tool docstring from TASK_TOOL_DESCRIPTION constant explicitly
# Use the constant directly as required by the user
SubagentToolKit.task.__doc__ = TASK_TOOL_DESCRIPTION
