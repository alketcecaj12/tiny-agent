"""Chapter 5 - Model Context Protocol (MCP).

Starts the bundled MCP server (`python -m tinyagent.mcp_server`) over stdio,
lists its tools and lets the agent use them.

Requires the optional dependency: pip install -e ".[mcp]"
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tinyagent import config
from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.mcp_tools import MCPTools
from tinyagent.memory import Memory

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "tinyagent.mcp_server"])
URL = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/refs/heads/main/README.md"


async def list_tools() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Name: {tool.name}")
                print(f"Description: {tool.description}")
                # `input_schema` in mcp 2.x, `inputSchema` in mcp 1.x
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
                print(f"Parameters: {schema}\n")


def main():
    asyncio.run(list_tools())

    # MCP-based tools are loaded automatically from the server
    tools = MCPTools()
    llm = LLM(model=config.model("prompt"))
    agent = TinyAgent(llm=llm, tools=tools, memory=Memory())
    print(agent.run(f"Read the file at '{URL}' and summarize it in three sentences."))


if __name__ == "__main__":
    main()
