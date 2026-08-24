---
name: tools-mcp-hitl-infrastructure
description: Use when working with Upsonic's tool layer that turns Python callables, ToolKits, MCP servers, and sub-agents into a uniform LLM-callable surface, including caching/retry/hook wrappers, HITL pause flows, the plan_and_execute orchestrator, deferred external execution, and shipped toolkits. Use when a user asks to register tools on an Agent, build a custom ToolKit, integrate an MCP server (stdio/SSE/streamable-HTTP), add human confirmation or user-input gates, defer tool execution to an external runner, configure caching/retries/timeouts via @tool, or use built-in provider tools and prebuilt integrations like Gmail, Slack, Telegram, Discord, WhatsApp, Tavily, DuckDuckGo, Exa, Firecrawl, Apify, Daytona, E2B, YFinance, BoCha, or generic SMTP/IMAP. Trigger when the user mentions ToolManager, ToolNormalizer, ToolRegistry, ToolWrapper, PauseHandler, OrchestratorLifecycle, NormalizationResult, ToolKit, Tool, ToolMetadata, ToolDefinition, ToolResult, ToolValidationError, ToolConfig, ToolHooks, FunctionTool, AgentTool, MCPTool, MCPHandler, MultiMCPHandler, FunctionSchema, ToolMetrics, plan_and_execute, Orchestrator, Thought, PlanStep, ConfirmationPause, UserInputPause, ExternalExecutionPause, PausedToolCall, UserControlFlowTools, UserInputField, @tool decorator, requires_confirmation, requires_user_input, external_execution, cache_results, tool_hooks, AbstractBuiltinTool, WebSearchTool, CodeExecutionTool, MCP, Model Context Protocol, HITL, or human-in-the-loop.
---

# `src/upsonic/tools/` — Tool, MCP, and HITL Infrastructure

This document describes the entire `src/upsonic/tools/` package: the central
plumbing that turns plain Python callables, decorated methods, third-party
SDK toolkits, MCP servers, and other agents into a uniform, behavior-rich,
LLM-callable tool surface for the Upsonic Agent runtime.

It also documents the framework's Human-In-The-Loop (HITL) mechanism, the
multi-step `plan_and_execute` orchestrator, the deferred/external execution
escape hatch, and every shipped tool wrapper.

---

## 1. What this folder is

`upsonic/tools/` is the **tool layer** that sits between user code (the things
you `tools=[...]` to an `Agent`) and the model loop (which expects a uniform
"name, JSON-schema parameters, executor" contract).

It owns five separable but cooperating concerns:

| Concern | Purpose | Files |
|---|---|---|
| Type system | Defines what a "tool" is across the codebase | `base.py`, `metrics.py`, `config.py`, `schema.py` |
| Registration | Walks user-supplied objects and produces uniform `Tool` instances | `normalizer.py`, `registry.py`, `__init__.py` (`ToolManager`) |
| Wrapping | Adds caching, retries, hooks, HITL pause points around raw tools | `execution.py` (`ToolWrapper`), `wrappers.py` |
| Multi-step planning | Treats `plan_and_execute` as a "meta-tool" the LLM emits to drive a loop | `orchestration.py` |
| Execution boundaries | Lets tool code **pause** the agent for confirmation, user input, or external execution | `hitl.py` (`PauseHandler` + the three `*Pause` exceptions), `user_input.py` |

In addition, two further concerns are anchored here:

| Concern | Purpose | Files |
|---|---|---|
| MCP integration | Connects Upsonic agents to Model Context Protocol servers (stdio / SSE / streamable-HTTP), prefixes tool names, surfaces remote tools as `MCPTool` | `mcp.py` |
| Built-in / shipped tools | Provider-side native tools (`WebSearchTool`, `CodeExecutionTool`, …), plus first-party Python toolkits for Gmail, Slack, Telegram, Daytona, E2B, Apify, Firecrawl, Exa, Tavily, DuckDuckGo, BoCha, YFinance, generic SMTP/IMAP, WhatsApp, Discord. | `builtin_tools.py`, `common_tools/`, `custom_tools/` |

The single public entry point for **everything** in this folder is the
`ToolManager` class (defined in `__init__.py`), which the `Agent` instantiates
once per agent and again per task.

---

## 2. Folder layout

```
src/upsonic/tools/
├── __init__.py              ← lazy re-exports + ToolManager (the high-level façade)
├── base.py                  ← Tool, ToolKit, ToolMetadata, ToolDefinition, ToolResult, ToolValidationError
├── config.py                ← ToolConfig, ToolHooks, @tool decorator
├── schema.py                ← FunctionSchema, function_schema(), GenerateToolJsonSchema
├── metrics.py               ← ToolMetrics
├── normalizer.py            ← ToolNormalizer, NormalizationResult
├── registry.py              ← ToolRegistry — owns all tool state + cascade-delete + instructions
├── execution.py             ← ToolWrapper — 9-aspect behavioral wrapper (cache/retry/hooks/HITL/...)
├── hitl.py                  ← PausedToolCall, ConfirmationPause / UserInputPause / ExternalExecutionPause, PauseHandler
├── wrappers.py              ← FunctionTool, AgentTool
├── orchestration.py         ← Thought / PlanStep / AnalysisResult / Orchestrator + plan_and_execute + OrchestratorLifecycle
├── user_input.py            ← UserInputField, UserControlFlowTools toolkit
├── mcp.py                   ← MCPTool, MCPHandler, MultiMCPHandler, transports
├── builtin_tools.py         ← AbstractBuiltinTool family + WebSearch / WebRead helpers
│
├── common_tools/            ← lightweight 1-function tools and selectable toolkits
│   ├── __init__.py
│   ├── tavily.py            ← tavily_search_tool() factory
│   ├── duckduckgo.py        ← duckduckgo_search_tool() factory
│   ├── bochasearch.py       ← bocha_search_tool() factory
│   └── financial_tools.py   ← YFinanceTools (uses .functions() protocol, not ToolKit)
│
└── custom_tools/            ← full ToolKit-based integrations (require API keys / SDKs)
    ├── __init__.py
    ├── apify.py             ← ApifyTools — dynamic Actor → tool registration
    ├── daytona.py           ← DaytonaTools — cloud sandbox (code, files, git)
    ├── discord.py           ← DiscordTools — Discord Bot REST API
    ├── e2b.py               ← E2BTools — E2B code interpreter sandbox
    ├── exa.py               ← ExaTools — neural web search + content
    ├── firecrawl.py         ← FirecrawlTools — scrape / crawl / map / extract
    ├── gmail.py             ← GmailTools — Gmail API via OAuth
    ├── mail.py              ← MailTools — generic SMTP + IMAP
    ├── slack.py             ← SlackTools — Slack Web/AsyncWeb client
    ├── telegram.py          ← TelegramTools — Telegram Bot API
    └── whatsapp.py          ← WhatsAppTools — Meta Graph API for WhatsApp
```

`__pycache__/` and an empty `custom_tools/__init__.py` round out the
filesystem; both are intentional and contain no logic.

### 2.1 Post-refactor architecture (TECH-1391)

The previous god class `ToolProcessor` and the dead-code
`DeferredExecutionManager` were retired. The tool layer is now five
small collaborators composed by `ToolManager`:

| Collaborator | File | Responsibility |
|---|---|---|
| `ToolNormalizer` | `normalizer.py` | Stateless type dispatch over the eight input kinds (raw function, bound method, `ToolKit` class/instance, tool-provider, agent instance, `MCPHandler`, plain class). Produces a `NormalizationResult` describing what to register. |
| `ToolRegistry` | `registry.py` | Owns every dict the legacy processor owned (`registered_tools`, `wrapped_tools`, `raw_object_ids`, MCP/class-instance ownership maps, KB/toolkit/provider instance maps). Provides cascade-delete over the eight input kinds and `collect_instructions` / `all_definitions`. |
| `ToolWrapper` | `execution.py` | Replaces the old `create_behavioral_wrapper`. Same 9-aspect pipeline (KB setup → before-hook → 3 pause checks → cache check → timeout-wrapped retry loop → metrics → cache write → show-result → after-hook → stop-after-call). Reads KB-related state from the `ToolRegistry` reference passed at construction. |
| `PauseHandler` | `hitl.py` | One place that turns any pause exception (`ConfirmationPause` / `UserInputPause` / `ExternalExecutionPause`) into a `PausedToolCall` and attaches it via `exc.paused_calls = [pc]`. `PausedToolCall` itself also lives here. |
| `OrchestratorLifecycle` | `orchestration.py` | Manages the `plan_and_execute` `Orchestrator` lifecycle around tool registration / removal (`maybe_create`, `update_context`, `maybe_discard`). |

`ToolManager` is now a ~150-line façade that wires these five together
and exposes `register_tools`, `remove_tools`, `execute_tool`,
`get_tool_definitions`, and `collect_instructions`. The five
collaborators are accessible as `tool_manager.normalizer`,
`.registry`, `.wrapper`, `.pause_handler`, and `.orchestrator_lifecycle`.

The HITL pause classes are exported from `upsonic.tools.hitl`. Code that
previously imported them from `upsonic.tools.processor` should switch
to `upsonic.tools.hitl`.

---

## 3. Top-level files

### 3.1 `base.py` — type primitives

This file defines the universal abstractions the rest of the package builds
on. It has no behavior; it only describes "what a tool is".

#### `ToolMetadata` (dataclass)

| Field | Type | Default | Role |
|---|---|---|---|
| `name` | `str` | required | Tool identifier shown to the LLM |
| `description` | `Optional[str]` | `None` | Human/LLM-readable description |
| `kind` | `Literal['function','output','external','unapproved','mcp']` | `'function'` | Discriminator used by `ToolDefinition.defer` etc. |
| `is_async` | `bool` | `False` | Whether the underlying implementation is async |
| `strict` | `bool` | `False` | Enforce strict JSON-schema validation on call args |
| `custom` | `Dict[str, Any]` | `{}` | Free-form bag (e.g. MCP server name, image attachments, execution history) |

#### `Tool` (abstract base class)

Every wrapped tool — function, agent, MCP, orchestrator — eventually inherits
from `Tool`. It owns:

- `name`, `description`, `schema`, `metadata`, `tool_id` (auto-derived from
  `f"{cls.__name__}_{name}"` if not supplied).
- A per-instance `_metrics: ToolMetrics`.
- `record_execution(execution_time, args, result, success)` which both bumps
  `metrics.tool_call_count` and appends to a 100-entry ring buffer stored
  under `metadata.custom['execution_history']`.
- `execute(*args, **kwargs)` — abstract; concrete classes must override.

```python
class Tool:
    def __init__(self, name, description=None, schema=None, metadata=None, tool_id=None):
        ...
        if tool_id is None:
            tool_id = f"{self.__class__.__name__}_{name}"
        ...
        self._metrics = ToolMetrics()

    @abstractmethod
    async def execute(self, *args, **kwargs): ...
```

#### `ToolKit` (config-only base class)

`ToolKit` is the base class **users** subclass when they want to ship a
collection of related tools (e.g. `SlackTools`, `GmailTools`). The class
itself does no discovery — discovery happens later, in
`ToolNormalizer._process_toolkit`. `ToolKit.__init__` only **stores**
configuration that the processor consumes:

- Filtering: `include_tools`, `exclude_tools` (lists of method names).
- Mode: `use_async` — if `True`, replaces the candidate set with all public
  async methods and drops every sync method (even `@tool`-decorated ones).
- Toolkit-wide instructions: `instructions`, `add_instructions`. When
  `add_instructions=True`, the text is injected into the agent's system
  prompt via `ToolRegistry.collect_instructions()`.
- Toolkit-wide HITL flag-lists: `requires_confirmation_tools`,
  `requires_user_input_tools`, `requires_external_execution_tools`.
- Toolkit-wide defaults that mirror **every** field of `ToolConfig`
  (`requires_confirmation`, `cache_results`, `cache_dir`, `cache_ttl`,
  `tool_hooks`, `max_retries`, `timeout`, `strict`, `docstring_format`,
  `require_parameter_descriptions`, etc.). These are stored in
  `_toolkit_defaults` and merged with the per-method `@tool` config.

After processing, `ToolNormalizer._process_toolkit` populates `self.tools`
with the wrapped callables; the `functions` property re-exposes that for
introspection.

```python
class MyKit(ToolKit):
    def __init__(self, api_key, **kw):
        super().__init__(**kw)
        self.api_key = api_key

    @tool(requires_confirmation=True)
    def dangerous(self, x: int) -> int:
        return x * 2
```

#### `ToolDefinition` (dataclass — outbound to the model)

```python
@dataclass
class ToolDefinition:
    name: str
    parameters_json_schema: Dict[str, Any] = {'type': 'object', 'properties': {}}
    description: Optional[str] = None
    kind: ToolKind = 'function'
    strict: Optional[bool] = None
    sequential: bool = False
    metadata: Optional[Dict[str, Any]] = None

    @property
    def defer(self) -> bool:
        return self.kind in ('external', 'unapproved')
```

This is the shape the agent ships to the model adapter. The model never sees
`Tool` instances — it sees `ToolDefinition`s.

#### `ToolResult` (dataclass — outbound from the executor)

```python
@dataclass
class ToolResult:
    tool_name: str
    content: Any
    tool_call_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
```

Returned by `ToolManager.execute_tool`; consumed by the agent loop to build
the next assistant message.

### 3.2 `config.py` — `ToolConfig`, `ToolHooks`, `@tool`

#### `ToolHooks`

A pydantic model with two optional fields, `before` and `after`. Both are
arbitrary callables, invoked synchronously around each tool execution by the
processor's behavioral wrapper. Their return values (if any) are stored under
`func_dict["func_before"]` / `func_dict["func_after"]` on the wrapper's
output.

#### `ToolConfig`

A pydantic model with the full set of behavioral flags. The decorator stores
this on the function object via `setattr(func, '_upsonic_tool_config', cfg)`.

| Field | Default | Behavior |
|---|---|---|
| `requires_confirmation` | `False` | Tool raises `ConfirmationPause` before executing — agent surfaces a paused call to the caller for approval. |
| `requires_user_input` | `False` | Tool raises `UserInputPause` before executing — caller must supply values for the listed `user_input_fields`. |
| `user_input_fields` | `[]` | Subset of parameter names the user fills in. The processor strips them from the JSON schema's `required` list so the LLM doesn't try to provide them. |
| `external_execution` | `False` | Tool raises `ExternalExecutionPause` — the host application executes it elsewhere and feeds the result back. |
| `show_result` | `False` | Print result to console but don't send back to the LLM. |
| `stop_after_tool_call` | `False` | After this tool, end the agent loop. |
| `sequential` | `False` | Force serial execution (no parallel tool dispatch). |
| `cache_results` | `False` | SHA-256-key cache to `cache_dir` with optional `cache_ttl`. |
| `cache_dir`, `cache_ttl` | `None`, `None` | Cache location / TTL in seconds. |
| `tool_hooks` | `None` | A `ToolHooks` instance. |
| `max_retries` | `5` | Retry on any non-pause exception with `2 ** attempt` backoff. Toolkits whose tools manage their own timeout/retry (e.g. `ShellToolKit.run_command` / `run_python` — subprocess-backed) override to `0` to avoid duplicating the wait and leaking child processes on cancel. |
| `timeout` | `30.0` | `asyncio.wait_for` timeout. `MCPTool` raises this to `60` (see the MCPTool default further below); subprocess-backed shell tools set it to `None` since `subprocess.run` / `asyncio.subprocess` owns its own timeout. |
| `strict` | `None` | Forwarded into `ToolMetadata.strict`. |
| `docstring_format` | `'auto'` | `'google' \| 'numpy' \| 'sphinx' \| 'auto'` — drives docstring parsing in `function_schema`. |
| `require_parameter_descriptions` | `False` | If `True`, schema generation fails when a parameter lacks a docstring description. |
| `instructions` | `None` | LLM-facing instructions auto-injected into the system prompt. |
| `add_instructions` | `False` | Gates whether the `instructions` are actually appended. |

A `model_validator` enforces that **at most one** of
`requires_confirmation`, `requires_user_input`, `external_execution` is active
per tool — the three HITL patterns are mutually exclusive.

#### `@tool` decorator

```python
@tool                                 # uses defaults
def f(x: int) -> int: ...

@tool(requires_confirmation=True,     # uses overrides
      cache_results=True, cache_ttl=300)
def g(x: int) -> int: ...
```

Internally, the decorator returns a `_ToolDecorator` helper that stores the
`ToolConfig` on the function as `_upsonic_tool_config` and marks it with
`_upsonic_is_tool = True`. These two attributes are the discovery contract
between `@tool` and `ToolNormalizer`.

### 3.3 `schema.py` — Pydantic-driven JSON-schema generation

This module converts a Python function (with type annotations and a parsed
docstring) into the JSON schema the LLM needs.

`function_schema(function, schema_generator, docstring_format, require_parameter_descriptions)` performs five validations:

1. The function must have a parseable signature.
2. It must have a non-empty docstring.
3. It must have a return-type annotation.
4. All parameters must have type annotations.
5. (Optional) All parameters must have docstring descriptions.

It then builds a Pydantic core schema, generates a `SchemaValidator`, and
runs `GenerateToolJsonSchema` to produce the public JSON schema. The result
is a `FunctionSchema` dataclass:

```python
@dataclass(kw_only=True)
class FunctionSchema:
    function: Callable[..., Any]
    description: str | None
    validator: SchemaValidator
    json_schema: dict[str, Any]
    is_async: bool
    single_arg_name: str | None = None
    positional_fields: list[str] = []
    var_positional_field: str | None = None

    async def call(self, args_dict: dict) -> Any: ...
```

`call()` correctly dispatches sync vs. async, runs sync tools in an
executor, and handles the "single model-like argument" optimization (where
a single `BaseModel` parameter is unwrapped at the top of the schema instead
of being nested under its parameter name).

`GenerateToolJsonSchema` is a small subclass of Pydantic's
`GenerateJsonSchema` that fixes two known issues: it correctly populates
`additionalProperties` for `TypedDict` types and removes useless property
titles. It is the schema generator passed everywhere a JSON schema is needed.

`SchemaGenerationError` wraps the aggregated validation errors with a
qualified function name in the message.

### 3.4 `metrics.py` — `ToolMetrics`

```python
@dataclasses.dataclass(repr=False, kw_only=True)
class ToolMetrics:
    tool_call_count: int = 0
    tool_call_limit: Optional[int] = None

    def can_call_tool(self) -> bool:
        return self.tool_call_limit is None or self.tool_call_count < self.tool_call_limit

    def increment_tool_count(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data) -> "ToolMetrics": ...
```

Each `Tool` carries its own `ToolMetrics` (`tool._metrics`) which is
incremented inside `Tool.record_execution`. It is also serialisable for
checkpoint persistence. There is no `to_json` — `dataclasses.asdict` is used
verbatim.

### 3.5 `normalizer.py` / `registry.py` / `execution.py` — the heart of the system

The work that the legacy `ToolProcessor` did is now split across three
focused collaborators:

1. **`ToolNormalizer` (`normalizer.py`)** dispatches raw user-supplied
   objects to a per-type processor (`_process_function_tool`,
   `_process_toolkit`, `_process_class_tools`, `_process_mcp_tool`,
   `_process_agent_tool`, `_process_tool_provider`), validates each via
   `function_schema`, and produces a `NormalizationResult` describing
   the new tools plus side-data (raw object IDs, MCP handler/owner
   maps, class-instance owners, KB / toolkit / provider instances).
   The normalizer is **stateless** — it never mutates a registry.
2. **`ToolRegistry` (`registry.py`)** owns every dict the legacy
   processor owned (`registered_tools`, `wrapped_tools`,
   `raw_object_ids`, `mcp_handlers`, `mcp_handler_to_tools`,
   `class_instance_to_tools`, `knowledge_base_instances`,
   `toolkit_instances`, `tool_provider_instances`). Its `add()` does
   the atomic merge of a `NormalizationResult`. Its `remove()`
   handles cascade-delete for every input kind (string, function,
   agent, MCP handler, ToolKit, tool provider, class, instance) and
   returns `(removed_tool_names, removed_original_objects)`.
3. **`ToolWrapper` (`execution.py`)** replaces `create_behavioral_wrapper`.
   Same 9-aspect pipeline (KB setup → before-hook → 3 pause checks →
   cache check → timeout-wrapped retry loop → metrics → cache write →
   show-result → after-hook → stop-after-call). It receives the
   `ToolRegistry` at construction so it can reach
   `knowledge_base_instances` and `class_instance_to_tools` for KB
   `setup_async()`.
6. **Collect instructions** (`collect_instructions`) for the system prompt.

Important branches inside `process_tools`:

```python
if isinstance(tool_item, Tool):                    return tool_item        # already wrapped
if self._is_builtin_tool(tool_item):               continue                # native, skip
if self._is_mcp_tool(tool_item):                   _process_mcp_tool(...)  # 1:N
if inspect.isfunction(tool_item):                  _process_function_tool
if inspect.ismethod(tool_item):                    _process_function_tool  # bound
if inspect.isclass(tool_item):
    if issubclass(tool_item, ToolKit):             _process_toolkit(tool_item())
    else:                                          _process_class_tools(tool_item())
elif isinstance(tool_item, ToolKit):               _process_toolkit
elif self._is_tool_provider(tool_item):            _process_tool_provider  # KnowledgeBase etc.
elif self._is_agent_instance(tool_item):           _process_agent_tool      # nested agent
else:                                              _process_class_tools     # plain instance
```

#### HITL pause exceptions

Three exception types live here. They are **not errors** — the agent loop
catches them and converts them into `PausedToolCall` payloads.

| Exception | Raised by | Caught by | Outcome |
|---|---|---|---|
| `ConfirmationPause` | behavioral wrapper when `config.requires_confirmation` | `ToolManager.execute_tool` | Builds a `PausedToolCall(requires_confirmation=True)` and re-raises so the agent can surface it. |
| `UserInputPause` | behavioral wrapper when `config.requires_user_input`, **or** raised manually by `UserControlFlowTools.get_user_input` for dynamic input | `ToolManager.execute_tool` | Builds the user-input schema (via `_build_user_input_schema_from_tool` or the schema attached to the exception) and creates a `PausedToolCall(requires_user_input=True, user_input_schema=...)`. |
| `ExternalExecutionPause` | behavioral wrapper when `config.external_execution` | `ToolManager.execute_tool` | Calls `PauseHandler.attach_paused_call` to mint a `PausedToolCall` and attach it to the exception, then re-raises. |

All three exceptions carry a `paused_calls: List[PausedToolCall]` attribute
that the agent loop reads in `agent.py` (lines ~2649-2695) to aggregate
multiple paused calls from a single LLM turn into one paused output.

#### Behavioral wrapper

`ToolWrapper(registry).wrap(tool)` returns an `async wrapper(**kwargs)`
closure that:

1. Triggers KnowledgeBase `setup_async()` lazily if the tool came from a KB.
2. Runs the `before` hook (stored as `func_before` if it returns).
3. Raises one of the three pauses **before** running the user function.
4. Looks up a SHA-256 cache key under `cache_dir/<key>.json` if
   `cache_results` is on, returning the hit immediately (with `func_cache`
   key set).
5. Runs the tool with `asyncio.wait_for(timeout)` and exponential-backoff
   retries up to `max_retries`. Pauses are never retried.
6. Calls `tool.record_execution(...)` for metrics.
7. Caches the result if requested.
8. Prints to console if `show_result`.
9. Runs the `after` hook.
10. Sets `_stop_execution=True` if `stop_after_tool_call` is on.

The return value is **not** the raw result — it is a `func_dict` with up to
five keys: `func`, `func_before`, `func_after`, `func_cache`,
`_stop_execution`. The agent's response processor consumes these.

#### Toolkit processing internals

`_process_toolkit(toolkit)` runs a two-phase algorithm:

**Phase 1 — discover candidates.** If `use_async=True`, scan **all** public
async methods. Otherwise, scan all bound methods that have
`_upsonic_is_tool=True` (set by `@tool`). Then add anything in
`include_tools`, then remove anything in `exclude_tools`.

**Phase 2 — merge configs and register.** For each candidate method:

- Start from the decorator's `ToolConfig` (or a fresh one).
- Apply `_apply_toolkit_config_overrides`, which lets toolkit-wide values
  in `_toolkit_defaults` (only those that are not `None`) **override** the
  decorator's values.
- Apply per-method HITL toolkit lists (`_requires_confirmation_tools`,
  `_requires_user_input_tools`, `_requires_external_execution_tools`).
- Wrap the bound method in a plain function via `_make_tool_wrapper` (which
  preserves `__self__` for cleanup later) and feed it through
  `_process_function_tool`.

After processing, `toolkit.tools` is populated with the wrapped callables
and `class_instance_to_tools[id(toolkit)]` is updated.

#### Tool provider protocol

Anything that exposes `get_tools() -> list[Tool | callable]` and is **not** a
`ToolKit` subclass is treated as a *tool provider*. `_process_tool_provider`
calls `provider.get_tools()`, registers the items (already-formed `Tool`s
go through verbatim, raw callables run through `_process_function_tool`),
and tracks the provider in `tool_provider_instances`. KnowledgeBase
instances are recognized here so their `setup_async()` is called lazily.

If a provider also implements `build_context() -> str`, that text is
collected into the system prompt by `collect_instructions()`.

#### `register_tools` vs. `process_tools`

`ToolNormalizer.normalize(tools, already_registered)` filters out any
tool whose `id(...)` was previously seen (via `ToolRegistry.raw_object_ids`)
and only processes the new ones — this is what `ToolManager.register_tools`
relies on so that re-registering the same agent tools or task tools
doesn't double-create wrappers.

### 3.6 `wrappers.py` — `FunctionTool` and `AgentTool`

#### `FunctionTool`

The standard `Tool` subclass for plain functions and bound methods.
Attributes: `function`, `config`, plus everything inherited from `Tool`.
`metadata.kind` is `'function'`.

It has two notable extras:

- `from_callable(c, name=None, description=None, config=None)` —
  one-step constructor that runs `function_schema` for you.  Tool providers
  (e.g. `KnowledgeBase`) prefer this over instantiating directly.
- `_convert_dicts_to_pydantic(kwargs)` — when the LLM gives you JSON for a
  parameter typed as `BaseModel` (or `Optional[Model]` or `List[Model]`),
  this method recursively validates and rebuilds the actual Pydantic
  model instance before calling the user function. It also handles
  `**kwargs`-style overflow parameters and reports detailed errors.

`execute(*args, **kwargs)` either awaits the function (if async) or runs
it via `loop.run_in_executor` (if sync), so blocking SDK calls never stall
the event loop.

#### `AgentTool`

Wraps any agent-like object so it can be passed in `tools=[...]`. The wrapped
object only needs `name`/`role`/`goal`/`system_prompt`/`do`/`do_async`. The
wrapper:

- Synthesizes a method name `ask_<sanitized_agent_name>`.
- Builds a description from the agent's role, goal, and system prompt.
- Generates a `FunctionSchema` from a stub `agent_function(request: str)`.
- Sets `metadata.kind = 'agent'`.
- On `execute(request, **kwargs)`, builds a `Task(description=request)` and
  calls `agent.do_async(task, return_output=True)` (or `agent.do(task)` in
  an executor as a fallback).
- If a parent agent has a `guardrail_provider`, wrappers created for child
  agents inherit it without mutating the child object. The inherited provider is
  resolved when the child is invoked, so reused child agents follow the current
  parent policy unless the child already defines its own provider.
- Accumulates child-agent token usage in `_accumulated_usage` (a `RunUsage`
  instance) and exposes it via `drain_accumulated_usage()` so the parent
  agent's run-output can roll up sub-agent costs.

### 3.7 `orchestration.py` — `plan_and_execute` and the `Orchestrator`

This file implements a "meta-tool": instead of calling each user tool one at
a time, the LLM emits a single `plan_and_execute(thought=...)` call carrying
a structured `Thought` (reasoning, plan, criticism, action). The
`Orchestrator` then executes the plan step-by-step.

#### Pydantic models

| Model | Fields | Role |
|---|---|---|
| `PlanStep` | `tool_name`, `parameters: Dict[str, Any]`, optional `description` | A single tool call inside a plan. |
| `Thought` | `reasoning`, `plan: List[PlanStep]`, `criticism`, `action: 'execute_plan' \| 'request_clarification'`, `clarification_needed?` | The structured "what do you intend to do" output the LLM emits. |
| `AnalysisResult` | `evaluation`, `next_action: 'continue_plan' \| 'revise_plan' \| 'final_answer'`, `reasoning?` | The orchestrator's mid-loop self-analysis. |
| `ExecutionResult` | `success`, `final_result`, `execution_history`, `total_steps`, `revisions` | Final synthesis result. |

#### `plan_and_execute` (the pseudo-tool)

```python
@tool(requires_confirmation=False, show_result=False,
      sequential=True, docstring_format='google')
def plan_and_execute(thought: Thought) -> str:
    """Master tool for complex tasks. Executes multi-step plans sequentially."""
    return "Plan received and will be executed by the orchestrator."
```

This is just the schema carrier. The actual execution is done by
`Orchestrator.execute(thought)`, wired up by `ToolManager.register_tools`
when `agent_instance.enable_thinking_tool` is true.

#### `Orchestrator`

A `Tool` subclass holding `agent_instance`, `task`, `wrapped_tools`, plus
loop state: `pending_plan`, `program_counter`, `revision_count`,
`execution_history` (a string log of every step).

Its `execute(thought)` runs:

```
while program_counter < len(pending_plan):
    step = pending_plan[program_counter]
    result = await _execute_single_step(step)
    if is_reasoning_enabled:
        # spin up a fresh tool-less Agent that returns AnalysisResult
        analysis = await _inject_analysis()
        if analysis.next_action == 'continue_plan':  program_counter += 1
        elif analysis.next_action == 'final_answer': break
        elif analysis.next_action == 'revise_plan':  await _request_plan_revision()
    else:
        program_counter += 1
return await _synthesize_final_answer()
```

Both the analysis and revision sub-agents are constructed in-line:

```python
analysis_agent = Agent(model=self.agent_instance.model,
                       name=f"{self.agent_instance.name}_analysis",
                       tools=[],            # explicitly no tools
                       enable_thinking_tool=False,
                       enable_reasoning_tool=False)
```

`_propagate_sub_agent_usage` rolls each sub-agent's `RunUsage` back into the
parent's `AgentRunOutput.usage` so token costs are not lost.

`is_reasoning_enabled` toggles whether to inject the analysis step at all
(controlled by `agent.enable_reasoning_tool`). When disabled, the
orchestrator just iterates straight through the plan.

### 3.8 `hitl.py` — `PausedToolCall`, pause exceptions, and `PauseHandler`

```python
@dataclass
class PausedToolCall:
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    requires_confirmation: bool = False
    requires_user_input: bool = False
    user_input_schema: Optional[List[Dict[str, Any]]] = None
    user_input_fields: Optional[List[str]] = None

    def to_dict(self) / from_dict(cls, data): ...   # serialisable
```

The three pause exception classes (`ConfirmationPause`, `UserInputPause`,
`ExternalExecutionPause`) and `PausedToolCall` all live here.
`PauseHandler.attach_paused_call(exc, *, tool_name, args, tool_call_id,
tool_obj)` is the single place that turns a raised pause exception into
a `PausedToolCall`, attaches it via `exc.paused_calls = [pc]`, and (for
`UserInputPause`) builds the `user_input_schema` from the tool's signature
when one isn't already supplied. The handler does **not** re-raise — the
caller (`ToolManager.execute_tool`) re-raises after attachment.

The `tools/user_input.py` lazy-import of `UserInputPause` and
`hitl.py`'s lazy-import of `_build_user_input_schema_from_tool` keep
the two modules from forming a circular import at module-load time;
this contract is pinned in the docstring at the top of `hitl.py`.

### 3.9 `user_input.py` — dynamic user input

#### `UserInputField` (dataclass)

`name`, `field_type` (the type's `__name__`), `description`, `value`. Used
both for static `requires_user_input=True` and dynamic
`UserControlFlowTools.get_user_input` flows.

`_build_user_input_schema_from_tool` reads the tool function's signature and
produces `UserInputField`s. Fields listed in `user_input_fields` get
`value=None` (must be filled by the user); other fields are pre-filled from
whatever the LLM passed in. If `user_input_fields` is empty, **all**
parameters require user input.

`_build_dynamic_user_input_schema(fields)` does the same for dynamic
LLM-emitted field lists.

#### `UserControlFlowTools` (toolkit)

A drop-in `ToolKit` that gives the agent a single tool, `get_user_input`,
which raises `UserInputPause` with a schema the LLM constructed itself:

```python
@tool
def get_user_input(self, fields: List[Dict[str, str]]) -> str:
    """fields: each dict has field_name, field_type, field_description"""
    schema = _build_dynamic_user_input_schema(fields)
    raise UserInputPause(user_input_schema=schema)
```

The toolkit's `_DEFAULT_INSTRUCTIONS` (a long, opinionated multi-line
string) is wired into `ToolKit.add_instructions=True` so it is automatically
appended to the system prompt. It tells the model to never ask for
information in plain text and to always call `get_user_input` instead.

### 3.10 `__init__.py` — public surface and `ToolManager`

The `__init__.py` is unusually rich. It does three things.

**1. Lazy re-exports.** `__getattr__(name)` walks a sequence of helper
functions (`_get_base_classes`, `_get_config_classes`, …) so that
`from upsonic.tools import FunctionTool` works without importing the whole
package up front (avoiding optional-dependency chains).

**2. Defines `ToolManager`.** The high-level façade composes the five
collaborators. Its surface:

| Method | Purpose |
|---|---|
| `__init__` | Instantiates `ToolNormalizer`, `ToolRegistry`, `ToolWrapper(registry=self.registry)`, `PauseHandler`, and `OrchestratorLifecycle(registry=self.registry)`. No tools are tracked yet; the orchestrator is created lazily when `plan_and_execute` is registered. |
| `register_tools(tools, task=None, agent_instance=None)` | Calls `normalizer.normalize(...)` to dedup and dispatch, `registry.add(result)` to merge atomically, `wrapper.wrap(tool)` for each new tool to install a behavioral wrapper in `registry.wrapped_tools`, then `orchestrator_lifecycle.maybe_create(...)` / `update_context(...)` to wire up `plan_and_execute` if it was newly registered. Returns the dict of newly registered tools. |
| `collect_instructions()` | Forwards to `registry.collect_instructions()`. |
| `_validate_required_args(tool_name, args)` | Pre-call sanity check. If the LLM truncated and left required args missing, returns a friendly error string (the agent then short-circuits with a failed `ToolResult`). |
| `execute_tool(tool_name, args, metrics=None, tool_call_id=None)` | Looks up the wrapped tool via `registry.get_wrapped(name)`. Validates required args; routes `plan_and_execute` to its orchestrator with a `Thought`; otherwise calls the wrapped tool with `**args`. Wraps the result in a `ToolResult`. On a pause exception, calls `pause_handler.attach_paused_call(exc, ...)` and re-raises. |
| `get_tool_definitions()` | Forwards to `registry.all_definitions()`. |
| `remove_tools(tools, registered_tools=None)` | Forwards to `registry.remove(tools)` and then `orchestrator_lifecycle.maybe_discard(removed_names)`. The registry handles all eight input kinds and distinguishes 1:1 (function/agent) from 1:N (MCP handler / ToolKit / tool provider) ownership. Returns `(removed_names, removed_original_objects)`. |

**3. Re-exports** `__all__` so `from upsonic.tools import X` works for every
public symbol (`Tool`, `ToolKit`, `ToolDefinition`, `ToolResult`,
`tool`, `ToolConfig`, `FunctionTool`, `AgentTool`, `Thought`,
`plan_and_execute`, `MCPHandler`, etc.).

---

## 4. Subfolders walked through

### 4.1 `mcp.py` — Model Context Protocol integration

Wraps the `mcp` Python SDK (an optional extra). The module is import-safe
even without `mcp` installed; it raises a clear ImportError only when
someone tries to instantiate `MCPHandler`/`MultiMCPHandler`.

#### Transports

| Transport | Params dataclass | Notes |
|---|---|---|
| stdio | `StdioServerParameters` (re-exported from `mcp.client.stdio`) | Default. Sandboxed by `prepare_command(...)` which rejects shell metacharacters and limits the executable to a hard-coded allowlist (`python`, `uvx`, `npx`, `node`, `docker`, …). |
| SSE | `SSEClientParams(url, headers?, timeout?, sse_read_timeout?)` | |
| Streamable HTTP | `StreamableHTTPClientParams(url, headers?, timeout?, sse_read_timeout?, terminate_on_close?, auth?)` | Builds a managed `httpx.AsyncClient` that the transport context manager closes on exit. |

#### `MCPTool`

A `Tool` subclass that wraps a single remote tool description. Its
`schema.json_schema` is the server's `inputSchema` (validator/function are
both `None` because the server validates server-side). `metadata.kind` is
`'mcp'` and `metadata.custom` includes `mcp_server`, `mcp_type`,
`mcp_transport`, `mcp_original_name`, and (if any) `mcp_tool_name_prefix`.
Default `ToolConfig`: `timeout=60`, `max_retries=2`, `sequential=False`.

`execute(**kwargs)` just delegates to `handler.call_tool(original_name, kwargs)`.

#### `MCPHandler`

| Method | Purpose |
|---|---|
| `__new__` | Issues a one-time security warning ("only trust MCP servers you connect to") and asserts the optional `mcp` package is installed. |
| `__init__(config=None, *, command=None, url=None, env=None, transport='stdio', server_params=None, session=None, timeout_seconds=5, include_tools=None, exclude_tools=None, tool_name_prefix=None)` | Accepts either a legacy config class (with `.url`/`.command`) or explicit kwargs, derives `connection_type`, `server_name`, and canonical `server_params`. |
| `connect()` | Async, idempotent. Opens the transport context manager, builds a `ClientSession`, calls `_discover_tools` (which calls `session.initialize()` and `session.list_tools()`), applies `include_tools`/`exclude_tools` filters, and creates one `MCPTool` per discovered tool. Guarded by an `asyncio.Lock`. |
| `close()` | Async. Tears down session and transport, calls `_managed_http_client.aclose()` if needed. Tool metadata survives. |
| `__aenter__` / `__aexit__` | Sugar for `connect`/`close`. |
| `get_tools()` | **Synchronous** wrapper that opens a temp connection in a fresh thread (if there's already a running loop) or a new event loop, discovers, then closes. Used at registration time, before the agent's main event loop starts. |
| `call_tool(tool_name, arguments)` | Auto-reconnects if the persistent session was closed, calls `session.call_tool`, processes the result via `_process_tool_result` (joins text content, base64-decodes embedded images, surfaces `EmbeddedResource` URIs, returns a dict with `content` + `images` when images were present). |
| `get_info()` | Returns a debug dict (server name, transport, tools, filters). |

`_cleanup_exception_handler` suppresses the well-known `RuntimeError("cancel
scope")` and "Task was destroyed but it is pending" warnings from `anyio`
when the transport is closed from a different task than the one that
opened it.

#### `MultiMCPHandler`

Coordinator for multiple MCP servers. Built either from a flat list of
`server_params_list`, a list of `commands`, or a list of `urls` (with an
optional aligned `urls_transports`). Each server gets its own `MCPHandler`
under the hood; tool name prefixes can be supplied either as a single
`tool_name_prefix` (which becomes `f"{prefix}_{idx}"` per server) or as a
parallel `tool_name_prefixes` list.

Aggregated `tools`, `handlers`. Same `connect`/`close`/`get_tools`/
async-context-manager surface as `MCPHandler`. Extras: `get_server_count()`,
`get_tool_count()`, `get_tools_by_server()` (dict server→tool names),
`get_server_info()` (per-server debug list).

### 4.2 `builtin_tools.py` — provider-side native tools

These are **declarative descriptions** of tools the model provider runs
itself — Anthropic web_search, OpenAI Responses code_execution, Google
url_context, etc. They never hit Python code at runtime. The agent layer
recognises them via `_is_builtin_tool` and forwards them to the provider in
`ModelRequestParameters` instead of going through the normal tool pipeline.

A `BUILTIN_TOOL_TYPES` registry is auto-populated by
`AbstractBuiltinTool.__init_subclass__`, keyed by the discriminator string
on the `kind` field. Pydantic discriminator-based deserialisation works
through `__get_pydantic_core_schema__`.

Shipped builtin types:

| Class | `kind` | Provider support |
|---|---|---|
| `WebSearchTool` | `'web_search'` | Anthropic, OpenAI Responses, Groq, Google, xAI. Optional `search_context_size`, `user_location`, `blocked_domains`/`allowed_domains`, `max_uses`. |
| `WebSearchUserLocation` (`TypedDict`) | n/a | Sub-payload for `WebSearchTool.user_location`. |
| `CodeExecutionTool` | `'code_execution'` | Anthropic, OpenAI Responses, Google, Bedrock Nova 2.0, xAI. |
| `WebFetchTool` | `'web_fetch'` | Anthropic, Google. Optional `max_uses`, allowed/blocked domains, `enable_citations`, `max_content_tokens`. |
| `UrlContextTool` (deprecated alias of `WebFetchTool`) | `'url_context'` | Backward-compatibility for old serialised payloads. |
| `ImageGenerationTool` | `'image_generation'` | OpenAI Responses, Google. Many optional fields (`background`, `quality`, `size`, `aspect_ratio`, `output_format`, …). |
| `MemoryTool` | `'memory'` | Anthropic. |
| `MCPServerTool` | `'mcp_server'` | OpenAI Responses, Anthropic, xAI. Carries `id`, `url`, `authorization_token`, `allowed_tools`, `headers`. |
| `FileSearchTool` | `'file_search'` | OpenAI Responses, Google. Carries `file_store_ids`. |

`DEPRECATED_BUILTIN_TOOLS = {UrlContextTool}`, `SUPPORTED_BUILTIN_TOOLS`
excludes deprecated, `BUILTIN_TOOLS_REQUIRING_CONFIG = {MCPServerTool,
MemoryTool}`.

The file also exports two **plain Python** helper functions:

- `WebSearch(query: str, max_results: int = 10) -> str` — uses `ddgs` /
  `duckduckgo_search` synchronously. Suitable as a quick ad-hoc tool.
- `WebRead(url: str) -> str` — uses `requests` + `bs4` to fetch and clean a
  page; truncates to 5000 chars. Both are used as zero-config defaults
  when an agent requests basic web access without a paid SDK.

### 4.3 `common_tools/` — light-weight shipped tools

`__init__.py` lazily re-exports `YFinanceTools`, `tavily_search_tool`,
`duckduckgo_search_tool`, `bocha_search_tool` so you can do
`from upsonic.tools.common_tools import tavily_search_tool`.

| File | Symbol | Style | What it does |
|---|---|---|---|
| `tavily.py` | `tavily_search_tool(api_key)` | factory returning a `@tool`-decorated async function | Wraps `tavily.AsyncTavilyClient` to expose `search(query, search_deep, topic, time_range)`; results validated through a `TavilySearchResult` `TypedDict` and a Pydantic `TypeAdapter`. |
| `duckduckgo.py` | `duckduckgo_search_tool(client=None, max_results=None)` | factory | Wraps `ddgs.DDGS` (or legacy `duckduckgo_search.DDGS`) on a thread, returns `List[DuckDuckGoResult]`. |
| `bochasearch.py` | `bocha_search_tool(api_key)` | factory | Calls Bocha AI's `/v1/web-search` endpoint with `aiohttp`, returns `List[BoChaSearchResult]`. |
| `financial_tools.py` | `YFinanceTools` | **non-`ToolKit`** class with a `functions()` method | Selectable bundle of yfinance accessors (`get_current_stock_price`, `get_company_info`, `get_analyst_recommendations`, `get_company_news`, `get_stock_fundamentals`, `get_income_statements`, `get_key_financial_ratios`, `get_historical_stock_prices`, `get_technical_indicators`). The `functions()` method exposes the user-selected subset; `process_tools` then handles the bound methods individually. |

The factory pattern (Tavily / DuckDuckGo / BoCha) is preferred over the
`ToolKit` pattern when there is a single tool that needs runtime config
(an API key) — the closure captures the key so the LLM never sees it.

### 4.4 `custom_tools/` — heavy-weight integrations

These are full `ToolKit` subclasses requiring a third-party SDK and usually
an API key. All follow the same structure:

```python
try:
    from <sdk> import <Client>
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

class XyzTools(ToolKit):
    def __init__(self, api_key=None, **kwargs):
        super().__init__(**kwargs)             # forward ToolKit options
        if not _AVAILABLE: import_error(...)   # friendly error
        self.api_key = api_key or getenv("XYZ_API_KEY", "")
        ...

    @tool
    def some_tool(self, ...) -> str: ...
    async def asome_tool(self, ...) -> str: ...   # async sibling, not @tool-decorated
```

Coverage at a glance:

| File | Class | SDK | Capability summary |
|---|---|---|---|
| `apify.py` | `ApifyTools(actors, apify_api_token, actor_defaults)` | `apify-client` | **Dynamic**: at construction time it fetches each Actor's input schema (`get_actor_latest_build`), prunes hidden fields, builds a real Python function with `inspect.Parameter`s for each visible property, and binds it as a method on the toolkit. The function carries `_json_schema_override` so `_process_function_tool` uses the rich Apify-provided schema instead of the auto-generated one. `actor_defaults` supplies fixed kwargs that are never shown to the LLM. |
| `daytona.py` | `DaytonaTools` | `daytona` | Lazy `Daytona` client + sandbox creation/connection. Tools: `daytona_run_code` (Python/TS/JS), `daytona_run_command`, `daytona_create_file`, `daytona_read_file`, `daytona_list_files`, `daytona_delete_file`, `daytona_install_packages`, `daytona_search_files`, `daytona_git_clone`, `daytona_get_sandbox_info`, `daytona_shutdown_sandbox`. Async siblings (`adaytona_*`) for each. |
| `discord.py` | `DiscordTools` | `httpx` (no SDK) | Discord Bot REST API: send messages with markdown / embeds / attachments / reactions / typing indicators, threads, DMs, pin/unpin, message edit/delete, guild/channel info. Uses `sanitize_text_for_discord` from `upsonic.utils.integrations.discord`. |
| `e2b.py` | `E2BTools` | `e2b-code-interpreter` | Lazy E2B `Sandbox`. Tools: code execution (Python, JS, Java, R, Bash), file upload/download, run shell, install packages, sandbox lifecycle (timeout, pause, resume, kill). |
| `exa.py` | `ExaTools` | `exa-py` | Tools: `search`, `find_similar`, `get_contents`, `answer`. Defaults for `num_results`, `search_type` (`auto`/`neural`/`keyword`), `text`, `highlights`, `summary`, plus domain/date filters. Uses `serialize_exa_response` helper. |
| `firecrawl.py` | `FirecrawlTools` | `firecrawl-py` | Tools: scrape single URL, crawl whole site (with depth/limits), map URLs, search the web with content scraping, batch scrape, LLM-extract structured data. Async job management via `start`/`status`/`cancel`. |
| `gmail.py` | `GmailTools` | `google-auth*`, `googleapiclient` | OAuth flow on first run (writes `token.json`). Tools: list/send/draft/read/search messages, threads, attachments. |
| `mail.py` | `MailTools` | stdlib | Generic SMTP+IMAP. IMAP tools: `get_unread_emails`, `get_latest_emails`, `get_emails_from_sender`, `search_emails`, `mark_email_as_read`/`unread`, `flag_email`/`unflag_email`, `delete_email`, `move_email`, `download_attachments`, `list_mailboxes`, `get_mailbox_status`. SMTP tools: `send_email`, `send_email_with_attachments`, `send_reply`, `send_reply_with_attachments`. Each tool has an `async def a<name>` sibling that delegates via `asyncio.to_thread`. |
| `slack.py` | `SlackTools` | `slack_sdk` | Sync + async clients. Tools: `send_message`, `send_message_thread`, `list_channels`, `get_channel_history`. Plus an `update_message` helper that is **not** marked `@tool`. |
| `telegram.py` | `TelegramTools` | `httpx` | Telegram Bot API: send text/photo/doc/audio/video/voice, inline & reply keyboards, chat actions, answer callback queries, file download, webhook management, message edit/delete. |
| `whatsapp.py` | `WhatsAppTools` | `httpx` | Meta Graph API. Tools: `send_text_message`, `send_template_message` (sync wrappers calling async siblings via `run_async`). |

These are illustrative implementations of the patterns in §3:
`@tool`-decorated methods, async siblings for the agent's async event loop,
optional-import guards with friendly `import_error` messages, and toolkit
defaults forwarded through `**kwargs` to `ToolKit.__init__`.

---

## 5. Cross-file relationships

The following diagram captures the registration → wrapping → execution →
HITL pipeline in roughly chronological order.

```
                           ┌──────────────────────────────────────────────────────────────┐
                           │  USER:  Agent(model=..., tools=[my_func, MyKit(), MCP(...)])  │
                           └──────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                          Agent.tool_manager = ToolManager()                       (agent.py:573)
                                          ToolManager.__init__():
                                              normalizer            = ToolNormalizer()
                                              registry              = ToolRegistry()
                                              wrapper               = ToolWrapper(registry=registry)
                                              pause_handler         = PauseHandler()
                                              orchestrator_lifecycle = OrchestratorLifecycle(registry=registry)
                                                       │
                                                       ▼
                                       Agent._setup_task_tools(task)   ──────►  ToolSetupStep        (pipeline/steps.py:463)
                                                       │
                                                       ▼
                                       ToolManager.register_tools(tools, task=, agent_instance=)
                                          ├─► normalizer.normalize(tools, registry.raw_object_ids)
                                          │     ├─ skip duplicates via raw_object_ids
                                          │     ├─ dispatch by type:
                                          │     │      function           → _process_function_tool          (FunctionTool)
                                          │     │      ToolKit            → _process_toolkit               (per-method @tool)
                                          │     │      class              → _process_class_tools
                                          │     │      tool_provider      → _process_tool_provider         (KnowledgeBase)
                                          │     │      MCPHandler         → _process_mcp_tool              (MCPTool×N)
                                          │     │      Agent              → _process_agent_tool            (AgentTool)
                                          │     │      AbstractBuiltinTool→ skip (handled by model adapter)
                                          │     └─ returns NormalizationResult (tools + side-data)
                                          │
                                          ├─► registry.add(result) — atomic merge into all state dicts
                                          │
                                          ├─► for each new tool: registry.store_wrapped(name, wrapper.wrap(tool))
                                          │   wraps: hooks → HITL pause checks → cache → timeout+retry
                                          │          → record_execution → cache write → show_result
                                          │          → after-hook → stop_after_tool_call
                                          │   stored in `registry.wrapped_tools[name]`
                                          │
                                          └─► orchestrator_lifecycle.maybe_create / update_context
                                              if 'plan_and_execute' newly registered AND agent.enable_thinking_tool:
                                                  build / refresh Orchestrator(agent_instance, task, wrapped_tools)
                                                  wrapped_tools['plan_and_execute'] = orchestrator_executor

                                                       │
                                                       ▼
                       ┌─────────────────────────────────────────────────────────────────┐
                       │  Agent loop (agent.py:_execute_tool_calls):                       │
                       │                                                                  │
                       │   for tool_call in llm_response.tool_calls:                      │
                       │      mgr = self._resolve_tool_manager(tool_call.tool_name)       │  (agent.py:2154)
                       │      try:                                                         │
                       │          result = await mgr.execute_tool(name, args, tool_call_id)│
                       │      except ConfirmationPause | UserInputPause                    │
                       │             | ExternalExecutionPause as e:                        │
                       │          aggregate e.paused_calls; pass back to caller           │
                       └─────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                          ToolManager.execute_tool(tool_name, args, tool_call_id):
                                  - _validate_required_args (LLM truncation guard)
                                  - if 'plan_and_execute': build Thought, call Orchestrator wrapper
                                  - else: await wrapped_tools[name](**args)
                                      └─ behavioral_wrapper:
                                              ConfirmationPause()    ◄─── if config.requires_confirmation
                                              UserInputPause()       ◄─── if config.requires_user_input
                                              ExternalExecutionPause()◄── if config.external_execution
                                              cache hit              ◄─── return without calling tool
                                              tool.execute(**kwargs) ◄─── happy path with retry+timeout
                                              record_execution()     ◄─── ToolMetrics + history ring buffer
                                  - on pause exception: PauseHandler.attach_paused_call(exc, ...)
                                    mints a PausedToolCall and attaches it via exc.paused_calls = [pc]; re-raise.
                                  - return ToolResult(...)
```

#### Where the three pause types travel

`ConfirmationPause`, `UserInputPause`, and `ExternalExecutionPause` are all
defined in `hitl.py`. They are raised inside the behavioral wrapper
**before** the tool function is called (so a `requires_confirmation` tool
never actually runs unless approved). `ToolManager.execute_tool` catches
them, attaches a `PausedToolCall` payload on `e.paused_calls`, and
re-raises. The agent loop in `agent.py` (around lines 2649-2695) then
gathers all paused calls from a single LLM turn and raises one
*aggregate* exception of the appropriate type, which the caller of
`agent.do_async` can intercept and respond to (with confirmation answers,
filled fields, or external results).

#### Knowledge-base lazy setup

If a tool comes from a `KnowledgeBase` (registered via the tool-provider
protocol), the behavioral wrapper calls `await kb.setup_async()` before the
first `execute(...)` so the underlying vector store is initialised exactly
once.

#### Toolkit instructions

`ToolRegistry.collect_instructions()` returns a deduplicated, ordered list
of instruction strings drawn from three sources:

1. `ToolKit` instances with `add_instructions=True` → `Instructions for toolkit «Name»: ...`.
2. Tool providers that implement `build_context()` (e.g. `KnowledgeBase`).
3. Individual `Tool`s whose `ToolConfig.add_instructions=True`.

`SystemPromptManager` (in `agent/context_managers/system_prompt_manager.py`)
calls this when assembling the final system prompt.

---

## 6. Public API

The following names are exported from `upsonic.tools` (`__init__.py`'s
`__all__`):

```python
# Type primitives
Tool, ToolKit, ToolDefinition, ToolResult, ToolMetadata
DocstringFormat, ObjectJsonSchema

# Configuration
tool, ToolConfig, ToolHooks

# Metrics
ToolMetrics

# Schema
FunctionSchema, function_schema, SchemaGenerationError

# Collaborators
ToolNormalizer, NormalizationResult
ToolRegistry
ToolWrapper
PauseHandler

# HITL pauses + paused-call dataclass
ConfirmationPause, UserInputPause, ExternalExecutionPause, PausedToolCall

# Validation
ToolValidationError

# Wrappers
FunctionTool, AgentTool

# Orchestration
PlanStep, AnalysisResult, Thought, ExecutionResult, plan_and_execute, Orchestrator, OrchestratorLifecycle

# MCP
MCPTool, MCPHandler, MultiMCPHandler
SSEClientParams, StreamableHTTPClientParams, prepare_command

# Manager
ToolManager

# Builtin tool descriptors
AbstractBuiltinTool, WebSearchTool, WebSearchUserLocation,
CodeExecutionTool, UrlContextTool, WebSearch, WebRead
```

Helpful but **not** in `__all__` (must be imported from their submodule):

```python
from upsonic.tools.hitl import ConfirmationPause, UserInputPause
from upsonic.tools.user_input import UserInputField, UserControlFlowTools
from upsonic.tools.builtin_tools import (
    WebFetchTool, ImageGenerationTool, MemoryTool, MCPServerTool, FileSearchTool,
    BUILTIN_TOOL_TYPES, SUPPORTED_BUILTIN_TOOLS, BUILTIN_TOOLS_REQUIRING_CONFIG,
)
```

---

## 7. Integration with the rest of Upsonic

### 7.1 `Agent.tool_manager`

In `src/upsonic/agent/agent.py:573`, every `Agent` constructs a
single `ToolManager`:

```python
self.tool_manager = ToolManager()
```

It then calls `self.tool_manager.register_tools(self.tools, agent_instance=self)`
in `_setup_agent_tools` (line 1912) to wire up agent-level tools at construction
time. Task-level tools are registered later, lazily, into a
**separate** `ToolManager` that lives on the task itself
(`task._tool_manager`, see §7.3).

The agent uses three key methods at runtime:

| Method | When called | What it gives the agent |
|---|---|---|
| `get_tool_definitions()` | `_run_pipeline` and `_execute_tool_calls` | The list of `ToolDefinition`s shipped to the model. Agent merges agent-level + task-level definitions before sending. |
| `execute_tool(name, args, tool_call_id)` | `_execute_tool_calls` for each `ToolCallPart` returned by the model | The `ToolResult` to feed back. May raise the three `*Pause` exceptions. |
| `remove_tools(tools, registered_tools)` | `Agent.remove_tool(...)` | Clean unregistration; returns the `(names, original_objects)` to delete from `agent.tools`. |

`Agent._resolve_tool_manager(tool_name)` (line 2154) decides whether a given
tool call should be routed to the agent's manager or the task's manager —
it prefers task-level registration when a name appears in both.

### 7.2 `ToolSetupStep` (pipeline)

`src/upsonic/agent/pipeline/steps.py:463` defines `ToolSetupStep`. It runs
sixth in the pipeline (`utils/pipeline.py:37`) and:

1. Calls `agent._setup_task_tools(task)` to merge agent-level + task-level
   tools.
2. Updates the `_planning_toolkit` with the current task if present.
3. Snapshots the combined tool definitions list and emits a streaming
   `tools_configured_event` (with `has_mcp_handlers` flag) for observers
   such as the langfuse exporter.

So `ToolSetupStep` is the place where the planning side of the tool layer
meets the rest of the pipeline; everything before this step is config, and
everything after assumes `tool_manager.get_tool_definitions()` is final.

### 7.3 Task-scoped tool managers

`Task._tool_manager: Optional[ToolManager]` (`tasks/tasks.py:81`,
properties at lines 507-518) is created lazily on first use via
`_ensure_tool_manager()`. This lets users do
`Task(description=..., tools=[extra_tool])` and have those tools be visible
**only** for that task's run. The agent then unions both managers'
definitions when calling the model and resolves the right manager when
executing a returned tool call.

Persistence: `Task.__getstate__` and the `data_to_task` deserialiser pickle
and unpickle `_tool_manager` so checkpointed tasks resume with their tool
state intact.

### 7.4 System prompt enrichment

`SystemPromptManager.build_full_system_prompt` calls
`agent.tool_manager.collect_instructions()` and (if present)
`task.tool_manager.collect_instructions()`, then appends the deduplicated
result to the system prompt. This is what carries `ToolKit.instructions`,
`ToolConfig.instructions`, and `KnowledgeBase.build_context()` through to
the LLM.

### 7.5 The `KnowledgeBase` ⇄ tools bridge

`KnowledgeBase` implements the *tool provider protocol* (it has
`get_tools()` returning a list of `Tool` objects, plus `setup_async()` and
`build_context()`). `ToolNormalizer._process_tool_provider` recognises it
specifically and adds the instance to both `tool_provider_instances` and
`knowledge_base_instances`. The behavioral wrapper then guarantees
`setup_async()` runs before the first KB tool execution.

---

## 8. End-to-end flow of a tool registration → call → execution

Below is the concrete trace for a single decorated function used inside an
agent run.

```python
# user code
from upsonic import Agent, Task
from upsonic.tools import tool, ToolHooks

@tool(cache_results=True, cache_ttl=300, max_retries=3,
      tool_hooks=ToolHooks(after=lambda r: print("done")))
def fetch_user(user_id: int) -> dict:
    """Look up a user by id.

    Args:
        user_id: The numeric user ID.
    """
    return external_db.get(user_id)

agent = Agent(model="openai/gpt-4o", tools=[fetch_user])
out = agent.do(Task("Fetch user 42"))
```

**Step 1 — Agent construction.**

`Agent.__init__` builds `self.tool_manager = ToolManager()`. Then
`_setup_agent_tools()` calls
`tool_manager.register_tools([fetch_user], agent_instance=self)`.

**Step 2 — Registration.**

```
ToolManager.register_tools
  ├─ normalizer.normalize([fetch_user], registry.raw_object_ids)
  │    ├─ id(fetch_user) ∉ raw_object_ids → kept
  │    └─ inspect.isfunction(fetch_user) → True
  │         _process_function_tool(fetch_user):
  │             config := fetch_user._upsonic_tool_config   (set by @tool)
  │             schema := function_schema(fetch_user, GenerateToolJsonSchema,
  │                                       'auto', False)
  │             tool_obj := FunctionTool(function=fetch_user,
  │                                      schema=schema, config=config)
  │             returns tool_obj
  │         result.tools = {'fetch_user': tool_obj}
  │
  ├─ registry.add(result)
  │    └─ registered_tools.update(result.tools)
  │       raw_object_ids.update(result.raw_object_ids)
  │
  └─ for 'fetch_user' in new_tools:
        registry.store_wrapped('fetch_user', wrapper.wrap(tool_obj))
```

`wrapped_tools['fetch_user']` is now an `async def wrapper(**kwargs)` with
the cache, retry, hook, pause, and metrics logic baked in.

**Step 3 — Pipeline.**

`ToolSetupStep` snapshots
`tool_manager.get_tool_definitions()` and constructs a `ToolDefinition`:

```python
ToolDefinition(
    name='fetch_user',
    description='Look up a user by id.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'user_id': {'type': 'integer', 'description': 'The numeric user ID.'}},
        'required': ['user_id'],
        'additionalProperties': False,
    },
    kind='function',
    strict=False,
    sequential=False,
    metadata=ToolMetadata(...),
)
```

That definition is shipped to the model adapter (OpenAI / Anthropic /
Bedrock / etc.) via `ModelRequestParameters`.

**Step 4 — Model emits a tool call.**

The model returns:

```json
{ "type": "tool_call", "name": "fetch_user", "args": { "user_id": 42 } }
```

**Step 5 — `Agent._execute_tool_calls` dispatches.**

It calls `mgr = self._resolve_tool_manager('fetch_user')` (returns the
agent-level manager since the task didn't override) and:

```python
result = await mgr.execute_tool('fetch_user',
                                args={'user_id': 42},
                                tool_call_id='call_<uuid>')
```

**Step 6 — `ToolManager.execute_tool`.**

```
- _validate_required_args('fetch_user', {'user_id': 42}) → None
- name != 'plan_and_execute', so:
- result = await wrapped_tools['fetch_user'](user_id=42)
   └ behavioral_wrapper:
        config := fetch_user_tool.config
        # before-hook absent
        # no requires_confirmation / requires_user_input / external_execution
        # cache_results=True → cache_key = sha256({"tool":"fetch_user","args":{"user_id":42}})
        # cache miss
        # execute with timeout=30s, retries=3
        result = await tool.execute(user_id=42)         # FunctionTool.execute
                  └ runs sync `fetch_user(42)` in loop.run_in_executor
        execution_time = ...
        tool.record_execution(...)                      # ToolMetrics++, history append
        cache_result(...)
        # show_result=False
        after-hook fires → 'done' printed → func_after captured
        return {"func": <the dict>, "func_after": ...}

- end:
- return ToolResult(tool_name='fetch_user', content=<func_dict>,
                    tool_call_id='call_<uuid>', success=True,
                    execution_time=...)
```

**Step 7 — Agent feeds the result back to the model**, the model emits the
final assistant message, and `agent.do` returns its output.

#### What changes if `requires_confirmation=True`

In step 6, `behavioral_wrapper` raises `ConfirmationPause()` **before**
calling `tool.execute`. The wrapper around it (in
`ToolManager.execute_tool`) catches the exception, builds a
`PausedToolCall(requires_confirmation=True, …)`, attaches it to
`e.paused_calls`, and re-raises. The agent loop aggregates pauses across
the whole turn and bubbles a single `ConfirmationPause` up to
`agent.do_async`. The caller then asks the user for approval, calls back
into the agent with the resolved decisions, and the loop re-runs
`execute_tool` with the same `tool_call_id` (now bypassing the
`requires_confirmation` check, since the agent layer has marked it
approved).

#### What changes if `requires_user_input=True`

Same shape, except `UserInputPause` is raised, the schema is built either
from `_build_user_input_schema_from_tool` (static fields supplied via
`user_input_fields`) or from the schema attached to the exception itself
(dynamic — `UserControlFlowTools.get_user_input` raises
`UserInputPause(user_input_schema=schema)` directly). The `PausedToolCall`
carries the field list; the host application fills in the values and
resumes.

#### What changes if `external_execution=True`

`ExternalExecutionPause` is raised. `ToolManager.execute_tool` calls
`deferred_manager.create_external_call(tool_name, args, tool_call_id)` to
mint a `PausedToolCall`, attach it, and re-raise. The host runs the call
itself (e.g. on another machine) and later calls
`deferred_manager.update_call_result(tool_call_id, result=...)`. The agent
re-enters the loop with the result already filled in.

#### What changes when `plan_and_execute` is registered

`ToolManager.register_tools` detects `'plan_and_execute' in newly_registered`
and `agent.enable_thinking_tool=True`, so it constructs an `Orchestrator`
holding the wrapped tools and registers `orchestrator_executor` as the
wrapper. When the LLM emits
`plan_and_execute(thought={reasoning, plan, criticism, action})`,
`ToolManager.execute_tool` recognises the special name, builds a `Thought`
from the args (handling both wrapped `{'thought': {...}}` and unwrapped
`{...}` shapes), and delegates to `Orchestrator.execute(thought)`. The
orchestrator then iterates through `pending_plan`, optionally injecting
analysis sub-agent calls between each step, and finally synthesises the
result using a third tool-less sub-agent. Token usage from each sub-agent
flows back into `agent_instance._agent_run_output.usage` via
`_propagate_sub_agent_usage`.

---

This concludes the tour. The folder's surface is large but its shape is
simple: `Tool` is the universal wrapper, `ToolNormalizer` discovers and
`ToolRegistry` tracks every tool, `ToolWrapper` adds the behavioral
pipeline, `PauseHandler` and `OrchestratorLifecycle` handle the two
side-channels (HITL pauses and `plan_and_execute`), and `ToolManager`
is the agent-facing façade composing the five together. MCP and the
prebuilt toolkits both flow through the same registration path, and
the three `*Pause` exceptions form a single cooperative HITL mechanism
that the agent loop knows how to drain into a single
`PausedToolCall` payload per turn.
