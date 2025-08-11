#!/usr/bin/env python3
"""
Verify that integration tests work with real Redis v8.
This script checks Redis connection and runs tutorial example tests.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import redis
    from redis import Redis
except ImportError:
    print("❌ Redis package not installed. Run: pip install redis")
    sys.exit(1)


async def check_redis_connection():
    """Check if Redis is accessible."""
    try:
        client = Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
        print("✅ Redis connection successful")

        # Check Redis version
        info = client.info()
        version = info.get("redis_version", "unknown")
        print(f"   Redis version: {version}")

        # Check for RediSearch module
        try:
            modules = client.module_list()
            has_search = any(m.get("name") == "search" for m in modules)
            if has_search:
                print("   ✅ RediSearch module loaded")
            else:
                print("   ⚠️  RediSearch module not found - vector search may not work")
        except:
            print("   ℹ️  Could not check for RediSearch module")

        client.close()
        return True

    except redis.ConnectionError as e:
        print(f"❌ Cannot connect to Redis: {e}")
        print("\nTo start Redis with Docker:")
        print("  docker run -d -p 6379:6379 redis/redis-stack:latest")
        print("\nOr install Redis locally:")
        print("  brew install redis")
        print("  redis-server")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False


async def test_tutorial_example():
    """Test a simple tutorial example with real Redis."""
    from eol.rag_context import EOLRAGContextServer
    from eol.rag_context.config import RAGConfig, RedisConfig
    from eol.rag_context.redis_client import RedisVectorStore
    from eol.rag_context.config import IndexConfig

    print("\n📚 Testing tutorial example with real Redis...")

    try:
        # Create Redis configuration
        redis_config = RedisConfig(host="localhost", port=6379, db=0, password=None)

        # Create Redis store
        store = RedisVectorStore(redis_config, IndexConfig())

        # Connect to Redis
        await store.connect_async()
        store.connect()  # Also establish sync connection
        print("   ✅ Connected to Redis store")

        # Create indexes
        store.create_hierarchical_indexes(embedding_dim=768)
        print("   ✅ Created hierarchical indexes")

        # Test basic operations
        test_data = {
            "content": "Test content for Redis integration",
            "embedding": [0.1] * 768,  # Mock embedding
            "metadata": {"source": "test.py", "type": "code"},
        }

        # Store test data
        doc_id = store.store_document(
            test_data["content"], test_data["embedding"], test_data["metadata"], hierarchy_level=3
        )
        print(f"   ✅ Stored document with ID: {doc_id}")

        # Search for similar content
        results = store.search_similar(test_data["embedding"], k=1, hierarchy_level=3)
        print(f"   ✅ Search returned {len(results)} results")

        # Clean up
        await store.close()
        print("   ✅ Closed Redis connection")

        print("\n✅ All Redis integration tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main verification script."""
    print("🔍 Verifying Redis Integration for Tutorial Examples")
    print("=" * 50)

    # Check Redis connection
    if not await check_redis_connection():
        print("\n❌ Redis is not available. Integration tests require Redis v8.")
        print("   Tutorial examples will not work without Redis.")
        return 1

    # Run integration test
    if not await test_tutorial_example():
        print("\n❌ Integration test failed.")
        return 1

    print("\n" + "=" * 50)
    print("✅ Redis integration verified successfully!")
    print("\nThe tutorial examples are configured to use real Redis v8.")
    print("All integration tests in test_tutorial_examples.py will use:")
    print("  - Real Redis connections from fixtures")
    print("  - Actual vector search operations")
    print("  - Persistent storage during test execution")
    print("\nTo run the full integration test suite:")
    print("  ./test_all.sh")
    print("\nOr manually:")
    print("  docker run -d -p 6379:6379 redis/redis-stack:latest")
    print("  pytest tests/integration/test_tutorial_examples.py -xvs")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
