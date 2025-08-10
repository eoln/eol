"""EOL MCP Server - Model Context Protocol server for EOL Framework"""

import os
import sys


def run_mcp_server():
    """Run EOL as an MCP server"""
    
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    
    if transport == "stdio":
        print("Starting EOL MCP server in stdio mode...", file=sys.stderr)
        # TODO: Implement stdio MCP server
    elif transport == "sse":
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8000"))
        print(f"Starting EOL MCP server on {host}:{port}...", file=sys.stderr)
        # TODO: Implement SSE MCP server
    else:
        print(f"Unknown transport: {transport}", file=sys.stderr)
        sys.exit(1)


__all__ = ["run_mcp_server"]