# gemini-server/server.py
import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Sequence

import google.generativeai as genai
from dotenv import load_dotenv
from mcp.server import McpError, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult, ErrorData, ImageContent, TextContent, Tool, EmbeddedResource,
    INVALID_PARAMS, INTERNAL_ERROR
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError  # Import ValidationError


# Load environment variables
load_dotenv()

# Gemini Configuration (retrieved dynamically)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Verbose mode for detailed output
verbose = True

async def generate_text(prompt: str):
    """Generates text using the configured Gemini model."""

    if verbose:
        print(f"\nPrompt sent to Gemini:\n{prompt}\n")

    model = genai.GenerativeModel(GEMINI_MODEL) # Dynamic Model configuration
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise McpError(INTERNAL_ERROR, f"Error generating text: {e}")

server = Server("gemini-server")

class GenerateArgs(BaseModel):
    """Input arguments for the 'generate' tool."""
    prompt: str = Field(..., description="Prompt to generate text from")
    model_config = ConfigDict(extra="allow")


@server.list_tools()  # This stays the same
async def list_tools() -> list[Tool]:
    """Lists available tools."""
    return [
        Tool(
            name="generate",
            description="Generates text using Gemini.",
            inputSchema={"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
        )
    ]


@server.call_tool() # Use @server.call_tool
async def call_tool(name: str, arguments: Dict) -> CallToolResult: # Correct signature
    """Handles tool calls for this server. Currently supports the 'generate' tool."""
    if name == "generate":
        try:
            args = GenerateArgs(**arguments) # Validation logic *moved* here
        except ValidationError as e:
            raise McpError(INVALID_PARAMS, f"Invalid 'generate' tool arguments: {e}")


        try:
            generated_text = await generate_text(args.prompt)
            return CallToolResult(content=[TextContent(type="text", text=generated_text)])
        except McpError as e:
            raise  # Re-raise MCP-specific errors
        except Exception as e:
            raise McpError(INTERNAL_ERROR, f"An unexpected error occurred: {e}")
    else:
        # If another tool name is requested, return an error
        raise McpError(INVALID_PARAMS, f"Unknown tool requested: {name}")




async def main():
    """Main function to start the server."""
    async with stdio_server() as streams:
        options = server.create_initialization_options()
        await server.run(*streams, options)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr) # Log to stderr
    asyncio.run(main())