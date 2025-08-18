#!/usr/bin/env python
"""Test the fixed MCP server components."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from our fixed launcher
sys.path.insert(0, str(Path(__file__).parent))
from mcp_launcher_fixed import indexer, initialize_components, redis_store


async def test_components():
    """Test all components work correctly."""
    print("Testing MCP Fixed Components")
    print("=" * 50)

    try:
        # Initialize
        result = await initialize_components()
        print(f"✅ Initialization: {result}")

        # Test get_stats (SYNC method)
        stats = indexer.get_stats()
        print(f"✅ Get stats (sync): {stats}")

        # Test list_sources (ASYNC method)
        sources = await indexer.list_sources()
        print(f"✅ List sources (async): {len(sources)} sources")

        # Test Redis connections
        print(f"✅ Sync Redis: {redis_store.redis is not None}")
        print(f"✅ Async Redis: {redis_store.async_redis is not None}")

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_components())
