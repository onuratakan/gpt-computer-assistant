"""Provider interfaces for pre-tool-call authorization.

These guardrails answer a different question than SafetyEngine policies:
SafetyEngine decides whether content is safe, while a GuardrailProvider decides
whether this agent is authorized to execute a specific tool call.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictBool

_LOGGER = logging.getLogger(__name__)


class GuardrailReason(BaseModel):
    """Structured explanation for an allow or deny decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    severity: str = "info"


class GuardrailDecision(BaseModel):
    """The result of a pre-tool-call authorization check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow: StrictBool
    reasons: tuple[GuardrailReason, ...] = Field(default_factory=tuple)
    provider_name: str | None = None

    @classmethod
    def allow_decision(
        cls,
        *,
        provider_name: str | None = None,
        reasons: Sequence[GuardrailReason] | None = None,
    ) -> "GuardrailDecision":
        return cls(allow=True, reasons=tuple(reasons or ()), provider_name=provider_name)

    @classmethod
    def deny_decision(
        cls,
        code: str,
        message: str,
        *,
        provider_name: str | None = None,
        severity: str = "error",
    ) -> "GuardrailDecision":
        return cls(
            allow=False,
            reasons=(GuardrailReason(code=code, message=message, severity=severity),),
            provider_name=provider_name,
        )

    def message(self) -> str:
        """Return a concise user-facing decision message."""

        if self.allow:
            return "Tool call authorized by guardrail provider."
        if not self.reasons:
            return "Tool call denied by guardrail provider."
        return "; ".join(reason.message for reason in self.reasons)


class GuardrailRequest(BaseModel):
    """Context passed to a GuardrailProvider before a tool executes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    arguments: Mapping[str, Any]
    tool_call_id: str | None = None
    tool_description: str = ""
    tool_parameters: Mapping[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    agent_id: str | None = None
    task_description: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @property
    def tool_input(self) -> Mapping[str, Any]:
        """Compatibility alias used by generic OAP-style providers."""

        return self.arguments


@runtime_checkable
class GuardrailProvider(Protocol):
    """Protocol for custom pre-tool-call authorization providers.

    Synchronous providers expose ``evaluate(request)``. Async providers should
    implement ``AsyncGuardrailProvider`` with ``aevaluate(request)``.
    """

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision | bool:
        ...


@runtime_checkable
class AsyncGuardrailProvider(Protocol):
    """Protocol for async pre-tool-call authorization providers."""

    name: str

    def aevaluate(self, request: GuardrailRequest) -> Awaitable[GuardrailDecision | bool]:
        ...


def evaluate_guardrail(
    provider: GuardrailProvider | AsyncGuardrailProvider,
    request: GuardrailRequest,
) -> GuardrailDecision:
    """Synchronously evaluate a provider and fail closed on provider errors."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aevaluate_guardrail(provider, request))
    raise RuntimeError("evaluate_guardrail() cannot run inside an event loop; use aevaluate_guardrail().")


async def aevaluate_guardrail(
    provider: GuardrailProvider | AsyncGuardrailProvider,
    request: GuardrailRequest,
) -> GuardrailDecision:
    """Evaluate a provider and fail closed on provider errors.

    Providers may expose either ``aevaluate(request)`` or ``evaluate(request)``.
    Returning ``bool`` is accepted for simple providers and normalized into a
    structured decision.
    """

    try:
        provider_name = _provider_name(provider)
        aevaluate = getattr(provider, "aevaluate", None)
        evaluate = getattr(provider, "evaluate", None)
        if aevaluate is not None:
            raw_decision = aevaluate(request)
        elif evaluate is not None:
            raw_decision = await asyncio.to_thread(evaluate, request)
        else:
            return GuardrailDecision.deny_decision(
                "guardrail.provider_missing_evaluator",
                "Guardrail provider must define evaluate() or aevaluate().",
                provider_name=provider_name,
            )

        if inspect.isawaitable(raw_decision):
            raw_decision = await raw_decision

        if isinstance(raw_decision, GuardrailDecision):
            if type(raw_decision.allow) is not bool:
                return GuardrailDecision.deny_decision(
                    "guardrail.invalid_decision",
                    "Guardrail provider returned a non-boolean allow value.",
                    provider_name=provider_name,
                )
            reasons = _normalize_reasons(raw_decision.reasons)
            if raw_decision.provider_name is None:
                return GuardrailDecision(
                    allow=raw_decision.allow,
                    reasons=reasons,
                    provider_name=provider_name,
                )
            return GuardrailDecision(
                allow=raw_decision.allow,
                reasons=reasons,
                provider_name=raw_decision.provider_name,
            )

        if isinstance(raw_decision, bool):
            if raw_decision:
                return GuardrailDecision.allow_decision(provider_name=provider_name)
            return GuardrailDecision.deny_decision(
                "guardrail.denied",
                "Tool call denied by guardrail provider.",
                provider_name=provider_name,
            )

        if hasattr(raw_decision, "allow"):
            allow = getattr(raw_decision, "allow")
            if type(allow) is not bool:
                return GuardrailDecision.deny_decision(
                    "guardrail.invalid_decision",
                    "Guardrail provider returned a non-boolean allow value.",
                    provider_name=provider_name,
                )
            return GuardrailDecision(
                allow=allow,
                reasons=_normalize_reasons(getattr(raw_decision, "reasons", ())),
                provider_name=getattr(raw_decision, "provider_name", None)
                or provider_name,
            )

        return GuardrailDecision.deny_decision(
            "guardrail.invalid_decision",
            "Guardrail provider returned an invalid decision.",
            provider_name=provider_name,
        )
    except Exception as exc:
        provider_name = locals().get("provider_name") or _safe_provider_class_name(provider)
        _LOGGER.warning(
            "Guardrail provider failed closed with %s.",
            type(exc).__name__,
        )
        return GuardrailDecision.deny_decision(
            "guardrail.provider_error",
            "Guardrail provider failed closed.",
            provider_name=provider_name,
        )


class AllowlistProvider:
    """Simple zero-dependency provider for allow/deny tool lists."""

    name = "allowlist"

    def __init__(
        self,
        *,
        allowed_tools: Sequence[str] | None = None,
        denied_tools: Sequence[str] | None = None,
    ) -> None:
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.denied_tools = set(denied_tools or ())

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name in self.denied_tools:
            return GuardrailDecision.deny_decision(
                "guardrail.tool_denied",
                f"Tool '{request.tool_name}' is denied by allowlist provider.",
                provider_name=self.name,
            )

        if self.allowed_tools is not None and request.tool_name not in self.allowed_tools:
            return GuardrailDecision.deny_decision(
                "guardrail.tool_not_allowed",
                f"Tool '{request.tool_name}' is not in the allowed tool list.",
                provider_name=self.name,
            )

        return GuardrailDecision.allow_decision(provider_name=self.name)


def _normalize_reasons(reasons: Iterable[Any]) -> tuple[GuardrailReason, ...]:
    if isinstance(reasons, GuardrailReason):
        reason_items = (reasons,)
    elif isinstance(reasons, Mapping) or isinstance(reasons, (str, bytes)):
        reason_items = (reasons,)
    else:
        try:
            reason_items = tuple(reasons or ())
        except TypeError:
            reason_items = (reasons,)

    normalized: list[GuardrailReason] = []
    for reason in reason_items:
        if isinstance(reason, GuardrailReason):
            normalized.append(
                GuardrailReason(
                    code=str(reason.code),
                    message=str(reason.message),
                    severity=str(reason.severity),
                )
            )
            continue
        if isinstance(reason, Mapping):
            normalized.append(
                GuardrailReason(
                    code=str(reason.get("code", "guardrail.reason")),
                    message=str(
                        reason.get(
                            "message",
                            reason.get("code", "Guardrail decision"),
                        )
                    ),
                    severity=str(reason.get("severity", "info")),
                )
            )
            continue
        code = getattr(reason, "code", None)
        message = getattr(reason, "message", None)
        severity = getattr(reason, "severity", "info")
        if code is None and message is None:
            message = str(reason)
        normalized.append(
            GuardrailReason(
                code=str(code or "guardrail.reason"),
                message=str(message or code or "Guardrail decision"),
                severity=str(severity),
            )
        )
    return tuple(normalized)


def _provider_name(provider: Any) -> str:
    try:
        name = getattr(provider, "name")
    except Exception:
        return _safe_provider_class_name(provider)
    if not isinstance(name, str) or not name:
        return _safe_provider_class_name(provider)
    return name


def _safe_provider_class_name(provider: Any) -> str:
    return type(provider).__name__ or "GuardrailProvider"
