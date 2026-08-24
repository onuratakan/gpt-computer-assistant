from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from upsonic.tools.metrics import ToolMetrics

@dataclass
class ToolExecution:
    """Execution of a tool"""

    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_error: Optional[bool] = None
    result: Optional[str] = None
    metrics: Optional[ToolMetrics] = None

    # In the case where a tool call creates a run of an agent/team/workflow
    child_run_id: Optional[str] = None

    # If True, the agent will stop executing after this tool call.
    stop_after_tool_call: bool = False

    created_at: int = field(default_factory=lambda: int(time()))

    # User control flow (HITL) fields
    requires_confirmation: Optional[bool] = None
    confirmed: Optional[bool] = None
    confirmation_note: Optional[str] = None

    requires_user_input: Optional[bool] = None
    user_input_schema: Optional[List[Dict[str, Any]]] = None
    answered: Optional[bool] = None

    external_execution_required: Optional[bool] = None

    result_injected: bool = False
    requires_guardrail_authorization: bool = False

    @property
    def is_paused(self) -> bool:
        return bool(self.requires_confirmation or self.requires_user_input or self.external_execution_required)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable values."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_call_error": self.tool_call_error,
            "result": self.result,
            "metrics": self.metrics.to_dict() if self.metrics and hasattr(self.metrics, 'to_dict') else None,
            "child_run_id": self.child_run_id,
            "stop_after_tool_call": self.stop_after_tool_call,
            "created_at": self.created_at,
            "requires_confirmation": self.requires_confirmation,
            "confirmed": self.confirmed,
            "confirmation_note": self.confirmation_note,
            "requires_user_input": self.requires_user_input,
            "user_input_schema": self.user_input_schema,
            "answered": self.answered,
            "external_execution_required": self.external_execution_required,
            "result_injected": self.result_injected,
            "requires_guardrail_authorization": self.requires_guardrail_authorization,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecution":
        """Reconstruct from dictionary."""
        metrics_data = data.get("metrics")
        metrics = None
        if isinstance(metrics_data, dict):
            metrics = ToolMetrics.from_dict(metrics_data)
        
        return cls(
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            tool_call_error=data.get("tool_call_error"),
            result=data.get("result"),
            child_run_id=data.get("child_run_id"),
            stop_after_tool_call=data.get("stop_after_tool_call", False),
            requires_confirmation=data.get("requires_confirmation"),
            confirmed=data.get("confirmed"),
            confirmation_note=data.get("confirmation_note"),
            requires_user_input=data.get("requires_user_input"),
            user_input_schema=data.get("user_input_schema"),
            answered=data.get("answered"),
            external_execution_required=data.get("external_execution_required"),
            result_injected=data.get("result_injected", False),
            requires_guardrail_authorization=data.get("requires_guardrail_authorization", False),
            metrics=metrics,
            created_at=data.get("created_at", int(time())),
        )
