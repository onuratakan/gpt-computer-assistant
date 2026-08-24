import logging
import threading

import pytest

from upsonic.guardrails import (
    AllowlistProvider,
    AsyncGuardrailProvider,
    GuardrailDecision,
    GuardrailProvider,
    GuardrailReason,
    GuardrailRequest,
    aevaluate_guardrail,
    evaluate_guardrail,
)
from tests.doc_examples.agent.guardrails.aport_verify_provider_example import (
    APortVerifyProvider,
)


def make_request(tool_name: str = "search") -> GuardrailRequest:
    return GuardrailRequest(tool_name=tool_name, arguments={"query": "hello"})


def test_guardrail_provider_protocol_accepts_named_sync_and_async_providers():
    assert isinstance(AllowlistProvider(allowed_tools=["search"]), GuardrailProvider)

    class AsyncOnlyProvider:
        name = "async_only"

        async def aevaluate(self, request):
            return True

    assert isinstance(AsyncOnlyProvider(), AsyncGuardrailProvider)


@pytest.mark.asyncio
async def test_allowlist_provider_allows_configured_tool():
    provider = AllowlistProvider(allowed_tools=["search"])

    decision = await aevaluate_guardrail(provider, make_request("search"))

    assert decision.allow is True
    assert decision.provider_name == "allowlist"


@pytest.mark.asyncio
async def test_allowlist_provider_denies_unlisted_tool():
    provider = AllowlistProvider(allowed_tools=["search"])

    decision = await aevaluate_guardrail(provider, make_request("write_file"))

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.tool_not_allowed"


@pytest.mark.asyncio
async def test_allowlist_provider_denied_tools_take_precedence():
    provider = AllowlistProvider(
        allowed_tools=["search", "write_file"],
        denied_tools=["write_file"],
    )

    decision = await aevaluate_guardrail(provider, make_request("write_file"))

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.tool_denied"


@pytest.mark.asyncio
async def test_evaluate_guardrail_accepts_bool_provider():
    class BoolProvider:
        name = "bool_provider"

        def evaluate(self, request):
            return request.tool_name == "search"

    allowed = await aevaluate_guardrail(BoolProvider(), make_request("search"))
    denied = await aevaluate_guardrail(BoolProvider(), make_request("write_file"))

    assert allowed.allow is True
    assert denied.allow is False
    assert denied.provider_name == "bool_provider"


def test_evaluate_guardrail_sync_entrypoint():
    provider = AllowlistProvider(allowed_tools=["search"])

    decision = evaluate_guardrail(provider, make_request("search"))

    assert decision.allow is True
    assert decision.provider_name == "allowlist"


@pytest.mark.asyncio
async def test_evaluate_guardrail_runs_sync_provider_off_event_loop():
    main_thread = threading.get_ident()
    captured = {}

    class SyncProvider:
        name = "sync_provider"

        def evaluate(self, request):
            captured["thread"] = threading.get_ident()
            return True

    decision = await aevaluate_guardrail(SyncProvider(), make_request())

    assert decision.allow is True
    assert captured["thread"] != main_thread


@pytest.mark.asyncio
async def test_evaluate_guardrail_accepts_async_provider():
    class AsyncProvider:
        name = "async_provider"

        async def aevaluate(self, request):
            return GuardrailDecision.allow_decision()

    decision = await aevaluate_guardrail(AsyncProvider(), make_request())

    assert decision.allow is True
    assert decision.provider_name == "async_provider"


@pytest.mark.asyncio
async def test_evaluate_guardrail_accepts_external_decision_object():
    class ExternalReason:
        code = "oap.tool_not_allowed"
        message = "Tool is not allowed by the passport."

    class ExternalDecision:
        allow = False
        reasons = [ExternalReason()]

    class ExternalProvider:
        name = "external_oap"

        def evaluate(self, request):
            return ExternalDecision()

    decision = await aevaluate_guardrail(ExternalProvider(), make_request("write_file"))

    assert decision.allow is False
    assert decision.provider_name == "external_oap"
    assert decision.reasons[0].code == "oap.tool_not_allowed"
    assert decision.reasons[0].message == "Tool is not allowed by the passport."


@pytest.mark.asyncio
async def test_evaluate_guardrail_rejects_non_bool_dataclass_allow():
    decision = await aevaluate_guardrail(
        object(),
        make_request(),
    )

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.provider_missing_evaluator"

    class InvalidProvider:
        name = "invalid_dataclass_allow"

        def evaluate(self, request):
            class InvalidDecision:
                allow = "false"
                reasons = ()

            return InvalidDecision()

    decision = await aevaluate_guardrail(InvalidProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.invalid_decision"
    assert "non-boolean allow" in decision.reasons[0].message


@pytest.mark.asyncio
async def test_evaluate_guardrail_normalizes_native_decision_reasons():
    class InvalidReasonProvider:
        name = "invalid_reason"

        def evaluate(self, request):
            class InvalidReasonDecision:
                allow = False
                reasons = ["denied"]

            return InvalidReasonDecision()

    decision = await aevaluate_guardrail(InvalidReasonProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.reason"
    assert decision.reasons[0].message == "denied"
    assert decision.message() == "denied"


@pytest.mark.asyncio
async def test_evaluate_guardrail_normalizes_native_reason_fields():
    class InvalidReasonFieldsProvider:
        name = "invalid_reason_fields"

        def evaluate(self, request):
            class InvalidReason:
                code = "policy.denied"
                message = 123

            class InvalidReasonDecision:
                allow = False
                reasons = [InvalidReason()]

            return InvalidReasonDecision()

    decision = await aevaluate_guardrail(InvalidReasonFieldsProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "policy.denied"
    assert decision.reasons[0].message == "123"
    assert decision.message() == "123"


@pytest.mark.asyncio
async def test_evaluate_guardrail_normalizes_native_mapping_reasons():
    class MappingReasonProvider:
        name = "mapping_reason"

        def evaluate(self, request):
            return GuardrailDecision(
                allow=False,
                reasons=[{"code": "policy.denied", "message": "Denied by policy"}],  # type: ignore[list-item]
            )

    decision = await aevaluate_guardrail(MappingReasonProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "policy.denied"
    assert decision.reasons[0].message == "Denied by policy"


@pytest.mark.asyncio
async def test_evaluate_guardrail_rejects_non_bool_external_allow():
    class ExternalDecision:
        allow = "false"
        reasons = []

    class ExternalProvider:
        name = "external_oap"

        def evaluate(self, request):
            return ExternalDecision()

    decision = await aevaluate_guardrail(ExternalProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.invalid_decision"
    assert "non-boolean allow" in decision.reasons[0].message


@pytest.mark.asyncio
async def test_evaluate_guardrail_sanitizes_provider_exception_details(caplog):
    caplog.set_level(logging.WARNING, logger="upsonic.guardrails.providers")

    class RaisingProvider:
        name = "raising"

        def evaluate(self, request):
            raise RuntimeError("internal endpoint https://api.internal.example token=secret")

    decision = await aevaluate_guardrail(RaisingProvider(), make_request())

    assert decision.allow is False
    assert decision.provider_name == "raising"
    assert decision.reasons[0].code == "guardrail.provider_error"
    assert decision.reasons[0].message == "Guardrail provider failed closed."
    assert "secret" not in decision.message()
    assert "api.internal" not in decision.message()
    assert "RuntimeError" in caplog.text
    assert "secret" not in caplog.text
    assert "api.internal" not in caplog.text


@pytest.mark.asyncio
async def test_evaluate_guardrail_handles_provider_name_property_failures():
    class BrokenNameProvider:
        @property
        def name(self):
            raise RuntimeError("name lookup failed")

        def evaluate(self, request):
            return True

    decision = await aevaluate_guardrail(BrokenNameProvider(), make_request())

    assert decision.allow is True
    assert decision.provider_name == "BrokenNameProvider"


def test_aport_verify_example_provider_uses_verify_api(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "decision": {
                    "allow": False,
                    "reasons": [
                        {
                            "code": "oap.tool_not_allowed",
                            "message": "Tool denied by APort",
                        }
                    ],
                }
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "tests.doc_examples.agent.guardrails.aport_verify_provider_example.requests.post",
        fake_post,
    )

    provider = APortVerifyProvider(
        agent_id="ap_test",
        api_key="apk_test",
        api_url="https://api.example.test",
    )

    decision = provider.evaluate(make_request("search_docs"))

    assert captured["url"] == "https://api.example.test/api/verify/policy/mcp.tool.execute.v1"
    assert captured["headers"]["Authorization"] == "Bearer apk_test"
    assert captured["json"]["context"]["agent_id"] == "ap_test"
    assert captured["json"]["context"]["tool"] == "search_docs"
    assert captured["json"]["context"]["parameters"] == {"query": "hello"}
    assert captured["timeout"] == 10.0
    assert decision.allow is False
    assert decision.reasons[0].code == "oap.tool_not_allowed"


def test_aport_verify_example_provider_rejects_invalid_allow(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"decision": {"allow": "false", "reasons": []}}

    monkeypatch.setattr(
        "tests.doc_examples.agent.guardrails.aport_verify_provider_example.requests.post",
        lambda *args, **kwargs: Response(),
    )

    provider = APortVerifyProvider(agent_id="ap_test", api_key="apk_test")

    decision = provider.evaluate(make_request("search_docs"))

    assert decision.allow is False
    assert decision.reasons[0].code == "oap.invalid_decision"


@pytest.mark.asyncio
async def test_evaluate_guardrail_fails_closed_on_provider_error():
    class BrokenProvider:
        name = "broken"

        def evaluate(self, request):
            raise RuntimeError("network unavailable")

    decision = await aevaluate_guardrail(BrokenProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.provider_error"
    assert decision.reasons[0].message == "Guardrail provider failed closed."
    assert "network unavailable" not in decision.message()


@pytest.mark.asyncio
async def test_evaluate_guardrail_fails_closed_on_invalid_decision():
    class InvalidProvider:
        name = "invalid"

        def evaluate(self, request):
            return {"allow": True}

    decision = await aevaluate_guardrail(InvalidProvider(), make_request())

    assert decision.allow is False
    assert decision.reasons[0].code == "guardrail.invalid_decision"
