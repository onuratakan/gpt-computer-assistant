"""HITL hub for the tools layer.

LAZY-IMPORT CONTRACT (do not break):
- This module MUST NOT import from ``tools.user_input`` at module load.
- ``PauseHandler.attach_paused_call`` imports ``_build_user_input_schema_from_tool``
  function-locally.
- ``tools.user_input:201`` imports ``UserInputPause`` from this module function-locally.
- Either side moving its import to module scope will create a circular import
  at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.tools.base import Tool
    from upsonic.tools.user_input import UserInputField


@dataclass
class PausedToolCall:
    """Represents a tool call paused for HITL handling (external execution, confirmation, or user input)."""

    tool_name: str
    """Name of the tool to execute."""

    tool_args: Dict[str, Any]
    """Arguments for the tool."""

    tool_call_id: str
    """Unique identifier for this tool call."""

    result: Optional[Any] = None
    """Result after external execution."""

    error: Optional[str] = None
    """Error message if execution failed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    requires_confirmation: bool = False
    """Whether this call requires user confirmation before execution."""

    requires_user_input: bool = False
    """Whether this call requires user-provided input values."""

    user_input_schema: Optional[List[Dict[str, Any]]] = None
    """Schema of fields the user must fill in (list of UserInputField dicts)."""

    user_input_fields: Optional[List[str]] = None
    """Subset of field names that the user must provide."""

    def args_as_dict(self) -> Dict[str, Any]:
        """Get arguments as dictionary."""
        return self.tool_args

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_call_id": self.tool_call_id,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "requires_confirmation": self.requires_confirmation,
            "requires_user_input": self.requires_user_input,
            "user_input_schema": self.user_input_schema,
            "user_input_fields": self.user_input_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PausedToolCall":
        """Reconstruct from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args", {}),
            tool_call_id=data["tool_call_id"],
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            requires_confirmation=data.get("requires_confirmation", False),
            requires_user_input=data.get("requires_user_input", False),
            user_input_schema=data.get("user_input_schema"),
            user_input_fields=data.get("user_input_fields"),
        )


class ExternalExecutionPause(Exception):
    """Exception to pause execution when external tool execution is required."""

    def __init__(self, paused_calls: List[PausedToolCall] = None):
        self.paused_calls: List[PausedToolCall] = paused_calls or []
        super().__init__(f"Paused for {len(self.paused_calls)} external tool calls")


class ConfirmationPause(Exception):
    """Exception to pause execution when user confirmation is required."""

    def __init__(self, paused_calls: List[PausedToolCall] = None):
        self.paused_calls: List[PausedToolCall] = paused_calls or []
        super().__init__(f"Paused for {len(self.paused_calls)} tool(s) requiring confirmation")


class UserInputPause(Exception):
    """Exception to pause execution when user input is required."""

    def __init__(
        self,
        paused_calls: List[PausedToolCall] = None,
        user_input_schema: Optional[List["UserInputField"]] = None,
    ):
        self.paused_calls: List[PausedToolCall] = paused_calls or []
        self.user_input_schema = user_input_schema or []
        super().__init__(f"Paused for {len(self.paused_calls)} tool(s) requiring user input")


class PauseHandler:
    """Single place that turns any pause exception into a ``PausedToolCall``
    and attaches it via ``exc.paused_calls = [pc]``.
    """

    def attach_paused_call(
        self,
        exc: Exception,
        *,
        tool_name: str,
        args: Dict[str, Any],
        tool_call_id: str,
        tool_obj: Optional["Tool"],
    ) -> None:
        """Attach a ``PausedToolCall`` to a pause exception.

        Sets ``exc.paused_calls = [pc]``. Does NOT re-raise — the caller
        is responsible for raising the exception after this returns.
        For ``UserInputPause`` it also sets ``exc.user_input_schema`` to
        the list of resolved ``UserInputField`` instances.
        """
        if getattr(exc, "paused_calls", None):
            return

        if isinstance(exc, ConfirmationPause):
            paused_call = PausedToolCall(
                tool_name=tool_name,
                tool_args=args,
                tool_call_id=tool_call_id,
                requires_confirmation=True,
            )
            exc.paused_calls = [paused_call]
            return

        if isinstance(exc, UserInputPause):
            if exc.user_input_schema:
                schema_fields = exc.user_input_schema
            else:
                config = getattr(tool_obj, 'config', None)
                user_input_fields_list: List[str] = (
                    config.user_input_fields if config else []
                )
                original_func = getattr(tool_obj, 'function', None)

                # Function-local import to satisfy the lazy-import contract
                # documented in this module's docstring.
                from upsonic.tools.user_input import _build_user_input_schema_from_tool
                schema_fields = _build_user_input_schema_from_tool(
                    tool_name=tool_name,
                    tool_args=args,
                    func=original_func,
                    user_input_fields=user_input_fields_list,
                )

            schema_dicts = [f.to_dict() for f in schema_fields]
            paused_call = PausedToolCall(
                tool_name=tool_name,
                tool_args=args,
                tool_call_id=tool_call_id,
                requires_user_input=True,
                user_input_schema=schema_dicts,
            )
            exc.paused_calls = [paused_call]
            exc.user_input_schema = schema_fields
            return

        if isinstance(exc, ExternalExecutionPause):
            paused_call = PausedToolCall(
                tool_name=tool_name,
                tool_args=args,
                tool_call_id=tool_call_id,
            )
            exc.paused_calls = [paused_call]
            return
