import asyncio
import json
import logging
import os
import sys
from typing import Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

async def main():
    # Point to the module1-server by relative path
    module1_server_path = "../module1-server/server.py"
    server_params = StdioServerParameters(command="python", args=[module1_server_path])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            goal = input("Enter your project goal: ")
            response = await session.call_tool("generate_plan", {"goal": goal})

            if response and response.get("content"):
                tool_result = response["content"][0].text if response.get("content") else None
                if tool_result is not None:
                    plan = json.loads(tool_result)
                    print("Generated plan:")
                    print(json.dumps(plan, indent=2))
                else:
                    print(f"No tool result available. Response was {response}")
            else:
                print("Invalid response from server:")
                print(response)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr) # Log to stderr
    asyncio.run(main())