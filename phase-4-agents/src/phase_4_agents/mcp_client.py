"""Model Context Protocol (MCP) Client and LangChain Adapter.

Provides helpers to:
1. Convert MCP server tools into LangChain `BaseTool` / `StructuredTool` instances,
   enabling seamless integration with LangGraph and LangChain agents.
2. Run an MCP client over standard I/O (`stdio`) to interact with MCP servers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPToolDefinition
from pydantic import BaseModel, create_model


def _sync_runner(coro: Any) -> Any:
    """Execute an async coroutine synchronously, handling running event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


def _json_schema_to_pydantic(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert an MCP tool's JSON input_schema into a dynamic Pydantic model."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for field_name, field_info in props.items():
        field_type: Any = Any
        if "type" in field_info and field_info["type"] in type_map:
            field_type = type_map[field_info["type"]]
        elif "anyOf" in field_info:
            for opt in field_info["anyOf"]:
                if opt.get("type") in type_map:
                    field_type = type_map[opt["type"]]
                    break
        default_val = field_info.get("default", ... if field_name in required else None)
        fields[field_name] = (field_type, default_val)

    return create_model(f"{name}_schema", **fields)


def convert_mcp_to_langchain_tools(server: MCPServer) -> list[BaseTool]:
    """Convert all tools exposed by an MCPServer into LangChain StructuredTools.

    Args:
        server: An active MCPServer instance.

    Returns:
        List of LangChain BaseTool objects ready for agent binding.
    """

    async def _get_tools() -> list[MCPToolDefinition]:
        return await server.list_tools()

    mcp_tools = _sync_runner(_get_tools())
    langchain_tools: list[BaseTool] = []

    for tool_def in mcp_tools:
        tool_name = tool_def.name
        tool_desc = tool_def.description or f"MCP tool: {tool_name}"
        input_schema = tool_def.input_schema or {}
        args_model = _json_schema_to_pydantic(tool_name, input_schema)

        # Factory to bind tool_name correctly in closure
        def make_callers(name: str):
            async def _async_call(**kwargs: Any) -> str:
                # Handle potential single-dict nesting from LangChain
                actual_args = kwargs
                if (
                    "kwargs" in kwargs
                    and len(kwargs) == 1
                    and isinstance(kwargs["kwargs"], dict)
                ):
                    actual_args = kwargs["kwargs"]

                res = await server.call_tool(name, actual_args)
                if isinstance(res, CallToolResult):
                    if res.content:
                        parts = [
                            c.text
                            for c in res.content
                            if isinstance(c, TextContent) and c.text is not None
                        ]
                        return "\n".join(parts)
                    if res.structured_content:
                        return str(res.structured_content)
                return ""

            def _sync_call(**kwargs: Any) -> str:
                return _sync_runner(_async_call(**kwargs))

            return _sync_call, _async_call

        sync_fn, async_fn = make_callers(tool_name)

        # Build StructuredTool with dynamic args_schema
        st = StructuredTool.from_function(
            func=sync_fn,
            coroutine=async_fn,
            name=tool_name,
            description=tool_desc,
            args_schema=args_model,
        )
        langchain_tools.append(st)

    return langchain_tools


@asynccontextmanager
async def open_mcp_stdio_client(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncGenerator[ClientSession, None]:
    """Connect to an external or local MCP server via stdio transport.

    Args:
        command: Executable command (e.g. 'python', 'uv').
        args: Command-line arguments to launch the MCP server.
        env: Optional environment variables dictionary.

    Yields:
        Initialized `ClientSession` ready to list tools, resources, and call tools.
    """
    server_params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session
