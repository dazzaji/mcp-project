import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import McpError, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    INTERNAL_ERROR,
    TextContent,
    Tool,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Verbose mode
verbose = True

# Define Pydantic model for plan structure validation
class PlanStructure(BaseModel):
    """Model for validating the initial plan structure."""

    Title: str
    Overall_Summary: str
    Original_Goal: str
    Detailed_Outline: List[Dict[str, str]] = Field(
        ..., description="List of steps with content"
    )
    Evaluation_Criteria: Dict[str, str] = Field(
        ..., description="Criteria for evaluating each step"
    )
    Success_Measures: List[str]

async def run_gemini_client(client: ClientSession, prompt: str) -> str:
    """Runs the Gemini client (via MCP) to generate the plan structure."""
    tools = await client.list_tools()

    if not any(tool.name == "generate" for tool in tools):
        raise RuntimeError("Gemini server does not support the 'generate' tool")

    response = await client.call_tool("generate", {"prompt": prompt})

    # Check for empty or invalid responses (add error code if desired)
    if response and response.get("content"):  # Check if response and content exist
        return response["content"][0].text # Only return the generated text, since result is inside content array

    # Handle empty or invalid responses
    raise McpError(INTERNAL_ERROR, f"Invalid response from Gemini server: {response}")

async def generate_plan_structure(goal: str) -> Optional[Dict]:
    """Generates the initial plan structure using Gemini Pro via MCP."""
    prompt = f"""
    You are a top consultant called in to deliver a final version of what the user needs correctly, completely, and at high quality.
    Create a comprehensive set of project deliverables, identifying each deliverable step by step, in JSON format to achieve the following goal: {goal}

    The JSON should strictly adhere to this template:
    {{
      "Title": "...",
      "Overall_Summary": "...",
      "Original_Goal": "{goal}",
      "Detailed_Outline": [
        {{"name": "Step 1", "content": "..."}},
        {{"name": "Step 2", "content": "..."}},
        ...
      ],
      "Evaluation_Criteria": {{
        "Step 1": "Criteria for Step 1",
        "Step 2": "Criteria for Step 2",
        ...
      }},
      "Success_Measures": ["...", "..."]
    }}

    Ensure that:
    1. Each step in the "Detailed_Outline" has a corresponding entry in the "Evaluation_Criteria"
    2. The Original_Goal field contains the exact goal provided
    3. Content is comprehensive but concise
    4. The response is valid JSON only, with no additional text or explanations
    """

    if verbose:
        print(f"\nPrompt for generating plan structure via MCP:\n{prompt}\n")

    #Gemini Server location (set dynamically or via env var)
    gemini_server_path = os.environ.get("GEMINI_SERVER_PATH") or "../gemini-server/server.py"

    async with stdio_client(StdioServerParameters(command="python", args=[gemini_server_path])) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                json_plan = await run_gemini_client(session, prompt)
                plan = json.loads(json_plan)
                try:
                    # Validate plan structure
                    validated_plan = PlanStructure(**plan)
                    return validated_plan.model_dump()
                except ValidationError as e:
                    raise McpError(INTERNAL_ERROR, f"Invalid plan structure: {str(e)}")
            except McpError as e:
                raise # Re-raise for client to handle
            except Exception as e:
                raise McpError(INTERNAL_ERROR, f"An unexpected error occurred: {e}")

async def main():
    """Main function to start the server."""
    server = Server("module1-server")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """Lists the available tools."""
        return [
            Tool(
                name="generate_plan",
                description="Generates a project plan structure from a goal.",
                inputSchema={
                    "type": "object",
                    "properties": {"goal": {"type": "string"}},
                    "required": ["goal"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict) -> CallToolResult:
        """Handles the 'generate_plan' tool call."""
        if name != "generate_plan":
            raise ValueError(f"Unknown tool: {name}")

        if not arguments or "goal" not in arguments:
            raise ValueError("Missing 'goal' argument.")
        goal = arguments["goal"]

        try:
            plan = await generate_plan_structure(goal)
            if plan:
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(plan, indent=2))]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text="Failed to generate plan.")
                    ],
                    isError=True,
                )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                isError=True,
            )

    async with stdio_server() as streams:
        options = server.create_initialization_options()
        await server.run(*streams, options)

if __name__ == "__main__":
    asyncio.run(main())
