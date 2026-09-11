# SPDX-License-Identifier: MIT
# Copyright 2026 AgentTanuki
"""Optional, unsigned observations of explicitly configured public endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool

_SERVICE = "https://agent-guild-5d5r.onrender.com/preflight"
_MAX_BYTES = 64 * 1024
_CHECKS = (
    "endpoint_reachable", "protocol_handshake", "agent_card_resolves",
    "agent_card_signed", "payment_claim_holds", "independent_evidence",
)
_STATUSES = frozenset(("proven", "failed", "unknown"))
_NOTICE = (
    "Unsigned, one-off Agent Guild observations. A reported signature presence "
    "is not signature verification. These checks do not establish ownership, "
    "competence, safety, payment completion, or future behavior. They neither "
    "authorize nor bind any later connection, delegation, data disclosure or payment."
)
_HEADERS = {
    "Accept": "application/json", "Accept-Encoding": "identity",
    "User-Agent": "AgentGuild-Upsonic/0.1.0",
}


class _ObservationError(ValueError):
    """A locally defined error code; never provider prose."""


def _public_url(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 2048
            or not value.isascii() or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
            or any(c in value for c in ("?", "#", "\\"))):
        raise _ObservationError("invalid_target")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise _ObservationError("invalid_target") from None
    if (parsed.scheme != "https" or not host or parsed.username is not None
            or parsed.password is not None or "%" in parsed.netloc or parsed.netloc.endswith(":")
            or (port is not None and not 1 <= port <= 65535)):
        raise _ObservationError("invalid_target")
    host = host.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if (len(labels) < 2 or len(host) > 253
                or not re.fullmatch(r"[a-z]{2,63}", labels[-1])
                or labels[-1] in {"localhost", "local", "internal", "invalid", "test"}
                or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)):
            raise _ObservationError("invalid_target") from None
    else:
        if not address.is_global or address.is_multicast:
            raise _ObservationError("invalid_target")
    # Syntax checks are not DNS/public-routing verification. The operator must
    # select a genuinely public URL, including a path free of secrets.
    return value


def _result_error(code: str) -> str:
    return json.dumps({"status": "unavailable", "error": code, "notice": _NOTICE})


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _ObservationError("invalid_response")
        result[key] = value
    return result


def _project(raw: bytes, target: str) -> str:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError):
        raise _ObservationError("invalid_response") from None
    if not isinstance(value, dict) or value.get("target") != target:
        raise _ObservationError("target_mismatch")
    checks = value.get("checks")
    if not isinstance(checks, list) or len(checks) != len(_CHECKS):
        raise _ObservationError("invalid_response")
    statuses: dict[str, str] = {}
    for item in checks:
        if not isinstance(item, dict):
            raise _ObservationError("invalid_response")
        name, status = item.get("check"), item.get("status")
        if (not isinstance(name, str) or name not in _CHECKS or name in statuses
                or not isinstance(status, str) or status not in _STATUSES):
            raise _ObservationError("invalid_response")
        statuses[name] = status
    projected = {
        "failed": [name for name in _CHECKS if statuses[name] == "failed"],
        "unknowns": [name for name in _CHECKS if statuses[name] == "unknown"],
        "scored": [name for name in _CHECKS if statuses[name] != "unknown"],
    }
    for key, expected in projected.items():
        supplied = value.get(key)
        if (not isinstance(supplied, list) or any(not isinstance(x, str) for x in supplied)
                or len(supplied) != len(set(supplied)) or set(supplied) != set(expected)):
            raise _ObservationError("inconsistent_response")
    # Provider detail, headline, verdict, names and instructions never enter output.
    return json.dumps({
        "status": "observed", "target": target,
        "checks": [{"check": name, "status": statuses[name]} for name in _CHECKS],
        **projected, "notice": _NOTICE,
    })


def _headers(response: httpx.Response) -> None:
    if response.status_code != 200:
        raise _ObservationError("service_http_error")
    if (response.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json"
            or response.headers.get("content-encoding", "identity").strip().lower() != "identity"):
        raise _ObservationError("unsupported_response")
    length = response.headers.get("content-length")
    if length is not None and (len(length) > 20 or not length.isdecimal() or int(length) > _MAX_BYTES):
        raise _ObservationError("response_size_limit")


def _append(data: bytearray, chunk: bytes, started: float, elapsed_limit: float) -> None:
    if time.monotonic() - started > elapsed_limit:
        raise _ObservationError("elapsed_limit")
    if len(data) + len(chunk) > _MAX_BYTES:
        raise _ObservationError("response_size_limit")
    data.extend(chunk)


def _fetch(target: str, io_timeout: float, elapsed_limit: float) -> str:
    started = time.monotonic()
    with httpx.Client(timeout=io_timeout, follow_redirects=False, trust_env=False, headers=_HEADERS) as client:
        with client.stream("GET", _SERVICE, params={"url": target}) as response:
            _headers(response)
            data = bytearray()
            for chunk in response.iter_raw(chunk_size=4096):
                _append(data, chunk, started, elapsed_limit)
            _append(data, b"", started, elapsed_limit)
    return _project(bytes(data), target)


async def _afetch(target: str, io_timeout: float, elapsed_limit: float) -> str:
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=io_timeout, follow_redirects=False, trust_env=False, headers=_HEADERS) as client:
        async with client.stream("GET", _SERVICE, params={"url": target}) as response:
            _headers(response)
            data = bytearray()
            async for chunk in response.aiter_raw(chunk_size=4096):
                _append(data, chunk, started, elapsed_limit)
            _append(data, b"", started, elapsed_limit)
    return _project(bytes(data), target)


class AgentGuildTools(ToolKit):
    """An opt-in observation tool, with no automatic framework interception.

    allowed_targets must be exact public HTTPS URLs selected by the operator.
    The entire selected URL is disclosed to Agent Guild, which actively probes
    it. Do not configure secrets in paths. No local DNS resolution is performed.

    io_timeout_seconds is an HTTPX per-I/O inactivity timeout, not a whole-call
    deadline. Sync elapsed limits are checked between blocking operations; DNS
    or an in-progress operation can overrun. Async also uses cooperative
    cancellation after elapsed_limit_seconds. Cancellation cannot undo a
    remote probe already requested. There are no retries.
    The HTTP helper starts no worker thread; Upsonic may execute sync tools in
    a worker, and cancelling that wrapper cannot stop the underlying sync operation.
    The framework timeout is elapsed_limit_seconds + io_timeout_seconds + 1.0.
    DNS or an active sync operation can outlive that fallback; framework timeout
    may produce a ToolResult failure while the underlying worker continues.
    """

    def __init__(
        self, allowed_targets: Sequence[str], *, use_async: bool = False,
        io_timeout_seconds: float = 10.0, elapsed_limit_seconds: float = 30.0,
    ) -> None:
        if isinstance(allowed_targets, (str, bytes)) or not allowed_targets:
            raise ValueError("Configure at least one explicit public endpoint URL.")
        for number in (io_timeout_seconds, elapsed_limit_seconds):
            if isinstance(number, bool) or not math.isfinite(number) or not 0 < number <= 60:
                raise ValueError("Timeout values must be finite, positive and at most 60 seconds.")
        self._allowed_targets = frozenset(_public_url(value) for value in allowed_targets)
        self._io_timeout = io_timeout_seconds
        self._elapsed_limit = elapsed_limit_seconds
        # No include_tools parameter: native inclusion is additive and could
        # otherwise expose arbitrary bound methods. Helpers live at module scope.
        super().__init__(
            use_async=use_async, add_instructions=False, cache_results=False,
            max_retries=0, timeout=elapsed_limit_seconds + io_timeout_seconds + 1.0,
        )

    @tool
    def observe_endpoint(self, url: str) -> str:
        """Request free unsigned observations for one configured public endpoint.

        Args:
            url: The exact operator-configured public HTTPS endpoint URL.

        Returns:
            JSON with measured check statuses and unknowns, or a local error code.
            This neither authorizes nor binds later endpoint execution.
        """
        try:
            target = _public_url(url)
            if target not in self._allowed_targets:
                return _result_error("target_not_configured")
            return _fetch(target, self._io_timeout, self._elapsed_limit)
        except _ObservationError as exc:
            return _result_error(str(exc))
        except httpx.TimeoutException:
            return _result_error("io_timeout")
        except httpx.HTTPError:
            return _result_error("transport_error")

    async def aobserve_endpoint(self, url: str) -> str:
        """Request free unsigned observations asynchronously.

        Args:
            url: The exact operator-configured public HTTPS endpoint URL.

        Returns:
            JSON with measured check statuses and unknowns, or a local error code.
            Cancellation closes the local operation; a remote probe may continue.
        """
        try:
            target = _public_url(url)
            if target not in self._allowed_targets:
                return _result_error("target_not_configured")
            return await asyncio.wait_for(
                _afetch(target, self._io_timeout, self._elapsed_limit),
                timeout=self._elapsed_limit,
            )
        except _ObservationError as exc:
            return _result_error(str(exc))
        except (asyncio.TimeoutError, httpx.TimeoutException):
            return _result_error("timeout")
        except httpx.HTTPError:
            return _result_error("transport_error")
