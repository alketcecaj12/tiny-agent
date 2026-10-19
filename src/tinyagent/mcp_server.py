"""A tiny MCP server exposing a `read_markdown` tool (Chapter 5, MCP).

Run it standalone with `python -m tinyagent.mcp_server`. Requires the optional
`mcp` dependency: `pip install -e ".[mcp]"`.
"""

try:  # newer `mcp` releases
    from mcp.server.mcpserver import MCPServer
except ImportError:  # `mcp` 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from tinyagent.toolbox import read_markdown as _read_markdown


# Initialize an MCP server
server = MCPServer("file_reader")


# Define our tools
@server.tool()
def read_markdown(path: str) -> str:
    """Read the content of a markdown file.

    Args:
        path (str): The URL or path to the markdown file.
    """
    return _read_markdown(path)


# Run the server
def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
