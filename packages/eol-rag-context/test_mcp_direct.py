#!/usr/bin/env python
"""Test MCP tools directly."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the fixed launcher module
sys.path.insert(0, str(Path(__file__).parent))
import mcp_launcher_fixed


async def test_tools():
    """Test MCP tools as they would be called from Claude."""
    print("Testing MCP Tools Directly")
    print("=" * 50)

    # Test get_stats tool (access the function directly)
    print("\n1. Testing get_stats()...")
    result = await mcp_launcher_fixed.get_stats.func()
    print(f"Result: {result}")

    # Test list_sources tool
    print("\n2. Testing list_sources()...")
    result = await mcp_launcher_fixed.list_sources.func()
    print(f"Result: {result}")

    # Test test_sandbox tool
    print("\n3. Testing test_sandbox()...")
    result = await mcp_launcher_fixed.test_sandbox.func()
    print(f"Result: {result}")

    print("\n✅ All tools executed successfully!")


if __name__ == "__main__":
    asyncio.run(test_tools())
