# Agent Guild endpoint observations

Use `AgentGuildTools` when an operator wants to expose a small set of public MCP or A2A endpoints for inspection. The toolkit sends one explicitly selected URL to Agent Guild's free `/preflight` service and returns a fixed set of check names, statuses and unknowns. It requires no Guild account, API key, registration or payment.

```python
from upsonic.tools.custom_tools import AgentGuildTools

# Replace this with a public endpoint you are authorized to have inspected.
endpoint = "https://example.com/mcp"
tools = AgentGuildTools(allowed_targets=[endpoint])
observations = tools.observe_endpoint(endpoint)
```

Pass this configured instance in an Upsonic task's `tools` list to expose `observe_endpoint`. Use `AgentGuildTools([endpoint], use_async=True)` to expose `aobserve_endpoint` instead, or call that method with `await` directly. Each mode registers one tool; the required argument is the exact configured `url`. An unlisted target returns `target_not_configured` without a request.

The entire selected URL is disclosed to `https://agent-guild-5d5r.onrender.com/preflight`, which actively probes it. Configure only public endpoints with no secrets in their paths. Query strings, fragments, embedded credentials and obvious local addresses are rejected. The client performs syntax checks, not DNS or routing verification. It sends no conversation, other tool arguments, credentials or payment material, and ignores environment proxy settings and redirects.

Output is JSON with controlled check names and statuses (`proven`, `failed`, `unknown`). Provider headlines, verdicts, free-form details and instructions are omitted. A positive protocol-handshake observation may mean an A2A card was found; it does not prove task execution. A signature-presence observation is not signature verification. Observations are unsigned and establish neither ownership nor competence, safety or payment completion. They do not authorize or bind later connections, delegation or execution.

Responses are limited to 64 KiB. By default, HTTPX uses a 10-second per-I/O inactivity timeout and the helper checks a 30-second elapsed limit between reads. These are not a hard wall-clock deadline for synchronous calls: DNS or an in-progress read may overrun. Upsonic may execute a synchronous tool in a worker thread; cancelling its wrapper does not terminate that operation. The asynchronous helper additionally uses cooperative cancellation at the elapsed limit. The toolkit sets Upsonic's outer timeout to the elapsed limit plus one I/O timeout plus one second (41 seconds by default). If that outer limit expires, the framework can return a failed tool result while the synchronous worker continues. Closing a local operation cannot undo a probe already requested from the service. No retries or result caching are configured by the toolkit.

Malformed, oversized, inconsistent, mismatched or unavailable responses produce a local error code. An unknown check remains unknown; it is not a failed safety test.

Validation used 21 tests and 28 subtests against the pinned Upsonic tool registration, schema and execution modules with mocked HTTP. It bypassed the top-level package initializer and did not invoke a model-backed agent, exercise Docker smoke tests or run the full project suite. The retained live-response fixture came from an owned service probe and is not evidence of independent adoption.
