"""MCP (Model Context Protocol) tool handling with comprehensive features."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import timedelta
from shlex import split as shlex_split
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Type, Union
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from upsonic.tools.base import Tool, ToolMetadata

# The `mcp` SDK is an optional extra (install with `pip install upsonic[mcp]`).
# We let the module import succeed without it so that downstream isinstance
# checks (e.g. processor.py) keep working; instantiating MCPHandler / MultiMCPHandler
# without the SDK raises a clear ImportError instead.
try:
    from mcp import types as mcp_types  # type: ignore[import-not-found]
    from mcp.client.session import ClientSession  # type: ignore[import-not-found]
    from mcp.client.sse import sse_client  # type: ignore[import-not-found]
    from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore[import-not-found]
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    mcp_types = None  # type: ignore[assignment]
    ClientSession = None  # type: ignore[assignment,misc]
    sse_client = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment,misc]

if _MCP_AVAILABLE:
    try:
        from mcp.client.streamable_http import streamable_http_client  # type: ignore[import-not-found]
        HAS_STREAMABLE_HTTP = True
    except ImportError:
        HAS_STREAMABLE_HTTP = False
        streamable_http_client = None  # type: ignore[assignment]
else:
    HAS_STREAMABLE_HTTP = False
    streamable_http_client = None  # type: ignore[assignment]


def _require_mcp() -> None:
    """Raise a friendly ImportError if the optional `mcp` SDK is missing."""
    if _MCP_AVAILABLE:
        return
    from upsonic.utils.printing import import_error
    import_error(
        package_name="mcp",
        install_command="pip install 'upsonic[mcp]'",
        feature_name="MCP (Model Context Protocol) tool integration",
    )


_MCP_SECURITY_WARNING_EMITTED = False


def _emit_mcp_security_warning() -> None:
    global _MCP_SECURITY_WARNING_EMITTED
    if _MCP_SECURITY_WARNING_EMITTED:
        return
    _MCP_SECURITY_WARNING_EMITTED = True

    from upsonic.utils.printing import console

    console.print(
        "[yellow]⚠️  MCP Security: Only connect to MCP servers you trust. "
        "Stdio servers run arbitrary processes on your machine; "
        "do not use commands or config from untrusted sources.[/yellow]"
    )


def prepare_command(command: str) -> List[str]:
    """
    Sanitize a command and split it into parts before using it to run an MCP server.
    
    Args:
        command: The command string to sanitize
        
    Returns:
        List of command parts safe for execution
        
    Raises:
        ValueError: If command contains dangerous characters or disallowed executables
    """
    DANGEROUS_CHARS = ["&", "|", ";", "`", "$", "(", ")"]
    if any(char in command for char in DANGEROUS_CHARS):
        raise ValueError(
            f"MCP command can't contain shell metacharacters: {', '.join(DANGEROUS_CHARS)}"
        )
    
    try:
        parts = shlex_split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not parts:
        raise ValueError("MCP command can't be empty")
    
    ALLOWED_COMMANDS = {
        "python", "python3", "uv", "uvx", "pipx",
        "node", "npm", "npx", "yarn", "pnpm", "bun",
        "deno", "java", "ruby", "docker",
    }
    
    first_part = parts[0]
    executable = first_part.split("/")[-1]
    
    if first_part.startswith(("./", "../")):
        return parts
    if first_part.startswith("/") and os.path.isfile(first_part):
        return parts
    if "/" not in first_part and os.path.isfile(first_part):
        return parts
    if shutil.which(first_part):
        return parts
    if executable not in ALLOWED_COMMANDS:
        raise ValueError(
            f"MCP command must use one of the following executables: {ALLOWED_COMMANDS}. "
            f"Got: '{executable}'"
        )
    
    return parts


@dataclass
class SSEClientParams:
    """Parameters for SSE (Server-Sent Events) client connection."""
    url: str
    headers: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = 5
    sse_read_timeout: Optional[float] = 60 * 5


@dataclass
class StreamableHTTPClientParams:
    """Parameters for Streamable HTTP client connection."""
    url: str
    headers: Optional[Dict[str, Any]] = None
    timeout: Optional[timedelta] = None
    sse_read_timeout: Optional[timedelta] = None
    terminate_on_close: Optional[bool] = None
    auth: Optional[Any] = None
    
    def __post_init__(self) -> None:
        if self.timeout is None:
            self.timeout = timedelta(seconds=30)
        if self.sse_read_timeout is None:
            self.sse_read_timeout = timedelta(seconds=60 * 5)

    def build_connect_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for ``streamable_http_client()``.
        
        Creates a managed httpx.AsyncClient internally so the transport
        context manager can close it properly on exit.
        """
        client_kwargs: Dict[str, Any] = {}
        if self.headers:
            client_kwargs["headers"] = self.headers
        if self.timeout:
            client_kwargs["timeout"] = httpx.Timeout(self.timeout.total_seconds())
        if self.auth:
            client_kwargs["auth"] = self.auth

        result: Dict[str, Any] = {"url": self.url}
        if client_kwargs:
            result["http_client"] = httpx.AsyncClient(**client_kwargs)
        if self.terminate_on_close is not None:
            result["terminate_on_close"] = self.terminate_on_close
        return result


class MCPTool(Tool):
    """Wrapper for MCP tools with enhanced capabilities."""
    
    def __init__(
        self,
        handler: 'MCPHandler',
        tool_info: mcp_types.Tool,
        tool_name_prefix: Optional[str] = None
    ):
        self.handler: MCPHandler = handler
        self.tool_info: mcp_types.Tool = tool_info
        self.original_name: str = tool_info.name
        self.tool_name_prefix: Optional[str] = tool_name_prefix
        
        if tool_name_prefix:
            prefixed_name = f"{tool_name_prefix}_{tool_info.name}"
        else:
            prefixed_name = tool_info.name
        
        from upsonic.tools.schema import FunctionSchema
        
        input_schema = tool_info.inputSchema if tool_info.inputSchema else {
            'type': 'object',
            'properties': {},
            'additionalProperties': True
        }
        
        mcp_schema = FunctionSchema(
            function=None,
            description=tool_info.description,
            validator=None,
            json_schema=input_schema,
            is_async=True,
            single_arg_name=None,
            positional_fields=[],
            var_positional_field=None
        )
        
        metadata = ToolMetadata(
            name=prefixed_name,
            description=tool_info.description,
            kind='mcp',
            is_async=True,
            strict=False
        )
        
        metadata.custom['mcp_server'] = handler.server_name
        metadata.custom['mcp_type'] = handler.connection_type
        metadata.custom['mcp_transport'] = handler.transport
        metadata.custom['mcp_original_name'] = self.original_name
        if tool_name_prefix:
            metadata.custom['mcp_tool_name_prefix'] = tool_name_prefix
        
        super().__init__(
            name=prefixed_name,
            description=tool_info.description,
            schema=mcp_schema,
            metadata=metadata
        )
        
        from upsonic.tools.config import ToolConfig
        self.config = ToolConfig(
            timeout=60,
            max_retries=2,
            sequential=False
        )
    
    async def execute(self, **kwargs: Any) -> Any:
        result = await self.handler.call_tool(self.original_name, kwargs)
        return result


class MCPHandler:
    """
    Handler for MCP server connections and tool management.
    
    Lifecycle:
        1. ``get_tools()`` (sync) – opens a temporary connection to discover
           available tools, then closes it.  Tool metadata is cached.
        2. ``call_tool()`` (async) – auto-reconnects on the caller's event
           loop if needed, then reuses the persistent session for every call.
        3. ``close()`` (async) – tears down the persistent connection.
           Tool metadata survives so the handler can be reconnected later.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "MCPHandler":
        _require_mcp()
        _emit_mcp_security_warning()
        return super().__new__(cls)

    def __init__(
        self,
        config: Type = None,
        *,
        command: Optional[str] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        server_params: Optional[Union[StdioServerParameters, SSEClientParams, StreamableHTTPClientParams]] = None,
        session: Optional[ClientSession] = None,
        timeout_seconds: int = 5,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
        tool_name_prefix: Optional[str] = None,
    ):
        self.session: Optional[ClientSession] = session
        self.tools: List[MCPTool] = []
        self.transport: str = transport
        self.timeout_seconds: int = timeout_seconds
        self.include_tools: Optional[List[str]] = include_tools
        self.exclude_tools: Optional[List[str]] = exclude_tools
        self.tool_name_prefix: Optional[str] = tool_name_prefix
        self._initialized: bool = False
        self._transport_ctx: Optional[Any] = None
        self._session_ctx: Optional[ClientSession] = None
        self._managed_http_client: Optional[httpx.AsyncClient] = None
        self._connect_lock: Optional[asyncio.Lock] = None
        
        if config is not None:
            if hasattr(config, 'url'):
                url = config.url
                transport = 'sse'
            elif hasattr(config, 'command'):
                cmd = config.command
                legacy_args = getattr(config, 'args', [])
                command = f"{cmd} {' '.join(str(arg) for arg in legacy_args)}" if legacy_args else cmd
                env = getattr(config, 'env', {})
                transport = 'stdio'
            else:
                raise ValueError("Config must have either 'url' or 'command' attribute")
            
            if tool_name_prefix is None and hasattr(config, 'tool_name_prefix'):
                self.tool_name_prefix = config.tool_name_prefix
        
        # --- Determine connection_type and server_name from inputs ---
        if server_params is not None:
            if isinstance(server_params, SSEClientParams):
                self.connection_type: str = 'sse'
                self.transport = 'sse'
                self.server_name: str = self._extract_server_name(server_params.url)
            elif isinstance(server_params, StreamableHTTPClientParams):
                if not HAS_STREAMABLE_HTTP:
                    from upsonic.utils.printing import import_error
                    import_error(
                        package_name="mcp[streamable-http]",
                        install_command="pip install 'mcp[streamable-http]'",
                        feature_name="MCP streamable HTTP transport"
                    )
                self.connection_type = 'streamable-http'
                self.transport = 'streamable-http'
                self.server_name = self._extract_server_name(server_params.url)
            elif isinstance(server_params, StdioServerParameters):
                self.connection_type = 'stdio'
                self.transport = 'stdio'
                self.server_name = server_params.command.split("/")[-1] if server_params.command else f"mcp_{uuid4().hex[:8]}"
            else:
                raise ValueError(f"Unsupported server_params type: {type(server_params)}")
        elif url:
            if transport == "sse":
                self.connection_type = 'sse'
            elif transport == "streamable-http":
                if not HAS_STREAMABLE_HTTP:
                    from upsonic.utils.printing import import_error
                    import_error(
                        package_name="mcp[streamable-http]",
                        install_command="pip install 'mcp[streamable-http]'",
                        feature_name="MCP streamable HTTP transport"
                    )
                self.connection_type = 'streamable-http'
            else:
                raise ValueError(f"Invalid transport for URL: {transport}")
            self.server_name = self._extract_server_name(url)
        elif command:
            self.connection_type = 'stdio'
            self.server_name = command.split()[0].split("/")[-1]
        else:
            raise ValueError("Must provide either url, command, or server_params")
        
        # --- Build canonical server_params ---
        if server_params:
            self.server_params: Union[StdioServerParameters, SSEClientParams, StreamableHTTPClientParams] = server_params
        elif self.connection_type == 'sse' and url:
            self.server_params = SSEClientParams(url=url)
        elif self.connection_type == 'streamable-http' and url:
            self.server_params = StreamableHTTPClientParams(url=url)
        elif self.connection_type == 'stdio' and command:
            parts = prepare_command(command)
            # Start from a minimal safe environment rather than the full
            # ``os.environ`` so API keys, cloud credentials, and other secrets
            # are not leaked to MCP server child processes (issue #621).
            from mcp.client.stdio import get_default_environment
            safe_env: Dict[str, str] = get_default_environment()
            if env is not None:
                safe_env.update(env)
            self.server_params = StdioServerParameters(
                command=parts[0],
                args=parts[1:] if len(parts) > 1 else [],
                env=safe_env
            )
        else:
            raise ValueError("Invalid configuration for MCP handler")
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_server_name(url: str) -> str:
        parsed = urlparse(url)
        return parsed.hostname or parsed.path.split('/')[-1] or 'mcp_server'
    
    def _build_transport_context(self) -> Any:
        """Build the raw transport context manager (not entered yet)."""
        if self.connection_type == 'sse':
            if isinstance(self.server_params, SSEClientParams):
                return sse_client(**asdict(self.server_params))
            return sse_client(url=str(getattr(self.server_params, 'url', self.server_params)))

        if self.connection_type == 'streamable-http':
            if not HAS_STREAMABLE_HTTP:
                raise ImportError("mcp[streamable-http] is required for streamable HTTP transport")
            if isinstance(self.server_params, StreamableHTTPClientParams):
                kwargs = self.server_params.build_connect_kwargs()
                self._managed_http_client = kwargs.get("http_client")
                return streamable_http_client(**kwargs)
            return streamable_http_client(url=str(getattr(self.server_params, 'url', self.server_params)))

        # stdio
        if not isinstance(self.server_params, StdioServerParameters):
            raise ValueError(f"stdio transport requires StdioServerParameters, got {type(self.server_params)}")
        return stdio_client(self.server_params)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _get_connect_lock(self) -> asyncio.Lock:
        """Lazily create the connection lock on first use.
        
        The lock must be created inside a running event loop, so it cannot
        live in ``__init__``.
        """
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        return self._connect_lock

    async def connect(self) -> None:
        """Open the transport, create a session, and discover tools.
        
        Serialised via an ``asyncio.Lock`` so parallel ``call_tool()``
        invocations don't race to create duplicate transports.
        """
        if self._initialized:
            return

        async with self._get_connect_lock():
            if self._initialized:
                return

            from upsonic.utils.printing import console

            if self.session is not None:
                await self._discover_tools()
                return
            
            console.print(f"[cyan]Connecting to MCP server: {self.server_name} ({self.connection_type})[/cyan]")
            
            try:
                self._transport_ctx = self._build_transport_context()
                transport_tuple = await self._transport_ctx.__aenter__()
                
                read, write = transport_tuple[0:2]
                
                read_timeout = max(self.timeout_seconds, 30)
                self._session_ctx = ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=read_timeout)
                )
                self.session = await self._session_ctx.__aenter__()
                
                await self._discover_tools()
                
                console.print(f"[green]✅ Connected to MCP server: {self.server_name}[/green]")
                
            except Exception as e:
                await self._force_cleanup()
                console.print(f"[red]❌ Failed to connect to MCP server: {e}[/red]")
                raise
    
    async def close(self) -> None:
        """Close the persistent connection. Tool metadata survives."""
        from upsonic.utils.printing import console
        
        await self._force_cleanup()
        
        console.print(f"[cyan]MCP handler for {self.server_name} closed[/cyan]")

    @staticmethod
    def _cleanup_exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
        """Suppress known harmless errors during cross-task transport cleanup.
        
        anyio raises RuntimeError('cancel scope') and asyncio warns about
        destroyed-but-pending tasks when we ``__aexit__`` a transport context
        manager from a different async task than the one that ``__aenter__``'d
        it.  Both are harmless because the underlying OS resources (pipes,
        sockets) are released regardless.
        """
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
            return
        msg = context.get("message", "")
        if "Task was destroyed" in msg:
            return
        loop.default_exception_handler(context)

    async def _force_cleanup(self) -> None:
        """Tear down session, transport, and any managed httpx clients."""
        loop = asyncio.get_event_loop()
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(self._cleanup_exception_handler)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
            try:
                if self._session_ctx is not None:
                    try:
                        await self._session_ctx.__aexit__(None, None, None)
                    except BaseException:
                        pass
                    self.session = None
                    self._session_ctx = None
                
                if self._transport_ctx is not None:
                    try:
                        await self._transport_ctx.__aexit__(None, None, None)
                    except BaseException:
                        pass
                    self._transport_ctx = None
                
                if self._managed_http_client is not None:
                    try:
                        await self._managed_http_client.aclose()
                    except BaseException:
                        pass
                    self._managed_http_client = None
                
                self._initialized = False
            finally:
                try:
                    await asyncio.sleep(0)
                except BaseException:
                    pass
                loop.set_exception_handler(original_handler)
    
    async def __aenter__(self) -> "MCPHandler":
        await self.connect()
        return self
    
    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[TracebackType]
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    async def _discover_tools(self) -> None:
        """Initialize session and discover tools (skips if tools already cached)."""
        if self._initialized:
            return
        
        from upsonic.utils.printing import console
        
        if not self.session:
            raise ValueError("Session not initialized")
        
        await self.session.initialize()
        
        if not self.tools:
            tools_response = await self.session.list_tools()
            
            available_tool_names = [tool.name for tool in tools_response.tools]
            self._validate_tool_filters(available_tool_names)
            filtered_tools = self._apply_tool_filters(tools_response.tools)
            
            console.print(
                f"[green]Found {len(filtered_tools)} tools from {self.server_name} "
                f"(total: {len(tools_response.tools)})[/green]"
            )
            
            self.tools = []
            for tool_info in filtered_tools:
                try:
                    tool = MCPTool(self, tool_info, tool_name_prefix=self.tool_name_prefix)
                    self.tools.append(tool)
                    if self.tool_name_prefix:
                        console.print(f"  - {tool.name} (original: {tool.original_name}): {tool.description}")
                    else:
                        console.print(f"  - {tool.name}: {tool.description}")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to register tool {tool_info.name}: {e}[/yellow]")
        
        self._initialized = True
    
    def _validate_tool_filters(self, available_tools: List[str]) -> None:
        if self.include_tools:
            invalid = set(self.include_tools) - set(available_tools)
            if invalid:
                raise ValueError(
                    f"include_tools references non-existent tools: {invalid}. "
                    f"Available tools: {available_tools}"
                )
        if self.exclude_tools:
            invalid = set(self.exclude_tools) - set(available_tools)
            if invalid:
                raise ValueError(
                    f"exclude_tools references non-existent tools: {invalid}. "
                    f"Available tools: {available_tools}"
                )
    
    def _apply_tool_filters(self, tools: List[mcp_types.Tool]) -> List[mcp_types.Tool]:
        filtered: List[mcp_types.Tool] = []
        for tool in tools:
            if self.exclude_tools and tool.name in self.exclude_tools:
                continue
            if self.include_tools is None or tool.name in self.include_tools:
                filtered.append(tool)
        return filtered

    # ------------------------------------------------------------------
    # Sync tool discovery (called before agent loop starts)
    # ------------------------------------------------------------------

    def get_tools(self) -> List[MCPTool]:
        """Discover tools synchronously. Opens a temp connection, discovers, closes it.
        
        The actual persistent connection is established lazily on the first
        ``call_tool()`` invocation inside the agent's async event loop.
        """
        from upsonic.utils.printing import console
        
        if self.tools:
            return self.tools
        
        async def _discover_and_close() -> List[MCPTool]:
            await self.connect()
            tools = list(self.tools)
            await self.close()
            return tools
        
        try:
            asyncio.get_running_loop()
            console.print("[yellow]⚠️  MCP async limitation detected. Attempting threaded connection...[/yellow]")
            
            import concurrent.futures
            
            def _run_in_thread() -> List[MCPTool]:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(_discover_and_close())
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_thread)
                self.tools = future.result(timeout=30)
            
            console.print("[green]✅ MCP tools discovered via thread[/green]")
            
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_discover_and_close())
            finally:
                loop.close()
        except Exception as e:
            console.print(f"[red]❌ MCP tool discovery failed: {e}[/red]")
            return []
        
        return self.tools

    # ------------------------------------------------------------------
    # Tool execution (runs inside the agent's async event loop)
    # ------------------------------------------------------------------

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server using the persistent session.
        
        Auto-reconnects if the session was closed (e.g. after ``get_tools()``
        discovery phase or a previous error).
        """
        from upsonic.utils.printing import console
        
        if not self._initialized or self.session is None:
            await self.connect()
        
        try:
            console.print(f"[blue]Calling MCP tool '{tool_name}' with args: {arguments}[/blue]")
            
            result: mcp_types.CallToolResult = await self.session.call_tool(tool_name, arguments)
            
            if result.isError:
                error_msg = f"Error from MCP tool '{tool_name}': {result.content}"
                console.print(f"[red]{error_msg}[/red]")
                return {"error": error_msg, "success": False}
            
            return self._process_tool_result(result, tool_name)
            
        except Exception as e:
            console.print(f"[red]Failed to call MCP tool '{tool_name}': {e}[/red]")
            raise

    # ------------------------------------------------------------------
    # Result processing
    # ------------------------------------------------------------------
    
    def _process_tool_result(self, result: mcp_types.CallToolResult, tool_name: str) -> Any:
        if not result.content:
            return None
        
        response_parts: List[str] = []
        images: List[Dict[str, Any]] = []
        
        for content_item in result.content:
            if isinstance(content_item, mcp_types.TextContent):
                text_content = content_item.text
                
                try:
                    parsed_json = json.loads(text_content)
                    if (
                        isinstance(parsed_json, dict)
                        and parsed_json.get("type") == "image"
                        and "data" in parsed_json
                    ):
                        image_data = parsed_json.get("data")
                        mime_type = parsed_json.get("mimeType", "image/png")
                        
                        if image_data and isinstance(image_data, str):
                            try:
                                image_bytes = base64.b64decode(image_data)
                                images.append({
                                    'id': str(uuid4()),
                                    'type': 'image',
                                    'content': image_bytes,
                                    'mime_type': mime_type,
                                    'source': 'mcp_custom_json'
                                })
                                response_parts.append("Image has been generated and added to the response.")
                                continue
                            except Exception:
                                pass
                except (json.JSONDecodeError, TypeError):
                    pass
                
                response_parts.append(text_content)
                
            elif isinstance(content_item, mcp_types.ImageContent):
                image_data = getattr(content_item, "data", None)
                
                if image_data and isinstance(image_data, str):
                    try:
                        image_bytes = base64.b64decode(image_data)
                    except Exception:
                        image_bytes = None
                else:
                    image_bytes = image_data
                
                images.append({
                    'id': str(uuid4()),
                    'type': 'image',
                    'url': getattr(content_item, "url", None),
                    'content': image_bytes,
                    'mime_type': getattr(content_item, "mimeType", "image/png"),
                    'source': 'mcp_image_content'
                })
                response_parts.append("Image has been generated and added to the response.")
                
            elif isinstance(content_item, mcp_types.EmbeddedResource):
                resource_info = {
                    'type': 'resource',
                    'uri': str(content_item.resource.uri),
                    'mime_type': getattr(content_item.resource, 'mimeType', None),
                    'text': getattr(content_item.resource, 'text', None)
                }
                response_parts.append(f"[Embedded resource: {json.dumps(resource_info)}]")
            
            else:
                response_parts.append(f"[Unsupported content type: {getattr(content_item, 'type', 'unknown')}]")
        
        response_text = "\n".join(response_parts).strip()
        
        if images:
            return {
                'content': response_text,
                'images': images,
                'success': True
            }
        return response_text if response_text else None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            'server_name': self.server_name,
            'connection_type': self.connection_type,
            'transport': self.transport,
            'tool_count': len(self.tools),
            'tools': [t.name for t in self.tools],
            'timeout_seconds': self.timeout_seconds,
            'initialized': self._initialized,
            'has_filters': bool(self.include_tools or self.exclude_tools),
            'tool_name_prefix': self.tool_name_prefix
        }
        if self.tool_name_prefix:
            info['original_tool_names'] = [t.original_name for t in self.tools]
        return info


class MultiMCPHandler:
    """
    Coordinator for managing multiple MCP server connections simultaneously.
    
    Creates one MCPHandler instance per server and aggregates their tools.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "MultiMCPHandler":
        _require_mcp()
        _emit_mcp_security_warning()
        return super().__new__(cls)

    def __init__(
        self,
        commands: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        urls_transports: Optional[List[Literal["sse", "streamable-http"]]] = None,
        *,
        env: Optional[Dict[str, str]] = None,
        server_params_list: Optional[
            List[Union[SSEClientParams, StdioServerParameters, StreamableHTTPClientParams]]
        ] = None,
        timeout_seconds: int = 5,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
        tool_name_prefix: Optional[str] = None,
        tool_name_prefixes: Optional[List[str]] = None,
    ):
        if server_params_list is None and commands is None and urls is None:
            raise ValueError("Must provide commands, urls, or server_params_list")
        
        self.timeout_seconds: int = timeout_seconds
        self.include_tools: Optional[List[str]] = include_tools
        self.exclude_tools: Optional[List[str]] = exclude_tools
        self.tool_name_prefix: Optional[str] = tool_name_prefix
        self.tool_name_prefixes: Optional[List[str]] = tool_name_prefixes
        self._initialized: bool = False
        self.tools: List[MCPTool] = []
        self.handlers: List[MCPHandler] = []
        
        self.server_params_list: List[Union[SSEClientParams, StdioServerParameters, StreamableHTTPClientParams]] = (
            server_params_list or []
        )
        
        inherited_env: Dict[str, str] = {**os.environ}
        if env is not None:
            inherited_env.update(env)
        env = inherited_env
        
        if commands:
            for command in commands:
                parts = prepare_command(command)
                self.server_params_list.append(
                    StdioServerParameters(
                        command=parts[0],
                        args=parts[1:] if len(parts) > 1 else [],
                        env=env
                    )
                )
        
        if urls:
            if urls_transports:
                if len(urls) != len(urls_transports):
                    raise ValueError("urls and urls_transports must be same length")
                for u, t in zip(urls, urls_transports):
                    if t == "streamable-http":
                        self.server_params_list.append(StreamableHTTPClientParams(url=u))
                    else:
                        self.server_params_list.append(SSEClientParams(url=u))
            else:
                for u in urls:
                    self.server_params_list.append(StreamableHTTPClientParams(url=u))
    
    async def connect(self) -> None:
        if self._initialized:
            return

        from upsonic.utils.printing import console

        console.print(f"[cyan]🔌 Connecting to {len(self.server_params_list)} MCP server(s)...[/cyan]")
        
        if self.tool_name_prefixes is not None:
            if len(self.tool_name_prefixes) != len(self.server_params_list):
                console.print(
                    f"[yellow]⚠️  tool_name_prefixes length ({len(self.tool_name_prefixes)}) does not match "
                    f"number of servers ({len(self.server_params_list)}). Skipping connection.[/yellow]"
                )
                self._initialized = True
                return
        
        for idx, server_params in enumerate(self.server_params_list):
            try:
                if self.tool_name_prefixes is not None:
                    prefix: Optional[str] = self.tool_name_prefixes[idx]
                elif self.tool_name_prefix is not None:
                    prefix = f"{self.tool_name_prefix}_{idx}"
                else:
                    prefix = None
                
                handler = MCPHandler(
                    server_params=server_params,
                    timeout_seconds=self.timeout_seconds,
                    include_tools=self.include_tools,
                    exclude_tools=self.exclude_tools,
                    tool_name_prefix=prefix
                )
                
                await handler.connect()
                
                self.handlers.append(handler)
                self.tools.extend(handler.tools)
                
                prefix_info = f" (prefix: {prefix})" if prefix else ""
                console.print(f"[green]  ✅ Server {idx+1}: {handler.server_name}{prefix_info} - {len(handler.tools)} tools[/green]")
                
            except Exception as e:
                console.print(f"[yellow]  ⚠️  Server {idx+1} connection failed: {e}[/yellow]")
        
        self._initialized = True
        console.print(f"[green]✅ Successfully connected to {len(self.handlers)} MCP servers with {len(self.tools)} total tools[/green]")
    
    async def close(self) -> None:
        for handler in self.handlers:
            try:
                await handler.close()
            except Exception:
                pass
        
        self.handlers.clear()
        self.tools.clear()
        self._initialized = False
    
    async def __aenter__(self) -> "MultiMCPHandler":
        await self.connect()
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.close()
    
    def get_tools(self) -> List[MCPTool]:
        from upsonic.utils.printing import console
        
        if self.tools:
            return self.tools
        
        async def _discover_and_close() -> List[MCPTool]:
            await self.connect()
            tools = list(self.tools)
            handlers = list(self.handlers)
            await self.close()
            self.tools = tools
            self.handlers = handlers
            return tools
        
        try:
            asyncio.get_running_loop()
            console.print("[yellow]⚠️  MCP async limitation detected. Attempting threaded connection...[/yellow]")
            
            import concurrent.futures
            
            def _run_in_thread() -> List[MCPTool]:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(_discover_and_close())
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_thread)
                self.tools = future.result(timeout=60)
            
            console.print("[green]✅ MCP tools discovered via thread[/green]")
            
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_discover_and_close())
            finally:
                loop.close()
        except Exception as e:
            console.print(f"[red]❌ MultiMCP tool discovery failed: {e}[/red]")
            return []
        
        return self.tools
    
    def get_server_count(self) -> int:
        return len(self.handlers)
    
    def get_tool_count(self) -> int:
        return len(self.tools)
    
    def get_tools_by_server(self) -> Dict[str, List[str]]:
        servers: Dict[str, List[str]] = {}
        for tool in self.tools:
            server_name: str = tool.metadata.custom.get('mcp_server', 'unknown')
            if server_name not in servers:
                servers[server_name] = []
            servers[server_name].append(tool.name)
        return servers
    
    def get_server_info(self) -> List[Dict[str, Any]]:
        info: List[Dict[str, Any]] = []
        for idx, handler in enumerate(self.handlers):
            handler_tools = [t for t in self.tools if t.handler == handler]
            server_info: Dict[str, Any] = {
                'index': idx,
                'server_name': handler.server_name,
                'connection_type': handler.connection_type,
                'transport': handler.transport,
                'tool_name_prefix': handler.tool_name_prefix,
                'tools': [t.name for t in handler_tools],
            }
            if handler.tool_name_prefix:
                server_info['original_tool_names'] = [t.original_name for t in handler_tools]
            info.append(server_info)
        return info
