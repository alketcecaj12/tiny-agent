"""Load tools from an MCP server (Chapter 5, MCP).

Requires the optional `mcp` dependency: `pip install -e ".[mcp]"`.
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tinyagent.tools import Tools


class MCPTools(Tools):
    """MCP-based tool registry that extends Tools."""

    def __init__(self, command: str | None = None, args: list[str] | None = None):
        """Connect to an MCP server started over stdio.

        By default this launches the bundled `tinyagent.mcp_server`.
        """
        super().__init__()
        self.command = command or sys.executable
        self.args = args if args is not None else ["-m", "tinyagent.mcp_server"]
        self._load_mcp_tools()

    def _get_server_params(self) -> StdioServerParameters:
        return StdioServerParameters(command=self.command, args=self.args)

    def _load_mcp_tools(self) -> None:
        """Fetch tools from MCP server and register them."""

        # Fetch the list of tools from the MCP server
        async def fetch():
            async with stdio_client(self._get_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    return tools.tools

        mcp_tools = asyncio.run(fetch())

        # Register each MCP tool using parent's add_tool
        for tool in mcp_tools:
            self.add_tool(
                name=tool.name,
                func=self._make_tool_caller(tool.name),
                description=tool.description,
            )

    def _make_tool_caller(self, name: str):
        """Create a callable that invokes the MCP tool."""

        def caller(**kwargs):
            return asyncio.run(self._call_tool(name, kwargs))

        return caller

    async def _call_tool(self, name: str, args: dict) -> str:
        """Call a tool on the MCP server."""
        async with stdio_client(self._get_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
                return result.content[0].text
