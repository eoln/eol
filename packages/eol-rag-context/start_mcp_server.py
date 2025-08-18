#!/usr/bin/env python
"""Start the EOL RAG Context MCP Server for local testing."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eol.rag_context.server import main

if __name__ == "__main__":
    print("Starting EOL RAG Context MCP Server...")
    print("=" * 50)
    print("To use with Claude Desktop, add this to your config:")
    print(
        """
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "python",
      "args": ["%s"]
    }
  }
}
"""
        % os.path.abspath(__file__)
    )
    print("=" * 50)

    # Run the server
    asyncio.run(main())
