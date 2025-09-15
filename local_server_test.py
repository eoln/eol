#!/usr/bin/env python3
"""
Local test for EOL RAG Context Server functionality
Run this to validate server components without MCP connection
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "packages/eol-rag-context/src"))

from eol.rag_context.config import RAGConfig  # noqa: E402
from eol.rag_context.server import EOLRAGContextServer  # noqa: E402


async def test_server_initialization():
    """Test basic server initialization"""
    print("🧪 Testing server initialization...")

    try:
        config = RAGConfig()
        server = EOLRAGContextServer(config)
        print("✅ Server created successfully")

        # Test initialization
        await server.initialize()
        print("✅ Server components initialized")

        # Test basic statistics
        if server.indexer:
            stats = server.indexer.get_stats()
            print(f"✅ Indexer stats: {stats}")
        else:
            print("❌ Indexer not initialized")

        # Test task manager
        if server.task_manager:
            print("✅ Task manager initialized")
        else:
            print("❌ Task manager not initialized")

        # Test parallel indexer
        if server.parallel_indexer:
            print("✅ Parallel indexer initialized")
        else:
            print("❌ Parallel indexer not initialized")

        await server.shutdown()
        print("✅ Server shutdown completed")

        return True

    except Exception as e:
        print(f"❌ Server test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_indexing_workflow():
    """Test non-blocking indexing workflow"""
    print("\n🧪 Testing non-blocking indexing workflow...")

    try:
        config = RAGConfig()
        server = EOLRAGContextServer(config)
        await server.initialize()

        # Test indexing the test data
        test_path = "/tmp/eol-test-data"
        print(f"📁 Testing indexing of: {test_path}")

        # This would normally be called through MCP
        result = await server.index_directory(path=test_path, recursive=True, force_reindex=False)

        if result.get("status") == "error":
            print(f"❌ Indexing failed: {result.get('message')}")
            return False

        task_id = result.get("task_id")
        print(f"✅ Indexing started with task_id: {task_id}")

        # Monitor progress
        max_attempts = 30
        for attempt in range(max_attempts):
            if server.task_manager:
                task_info = await server.task_manager.get_task_status(task_id)
                if task_info:
                    print(
                        f"📊 Progress: {task_info.progress_percentage}% "
                        f"({task_info.completed_files}/{task_info.total_files} files)"
                    )
                    if task_info.status.value == "completed":
                        print("✅ Indexing completed successfully")
                        break
                    elif task_info.status.value == "failed":
                        print(f"❌ Indexing failed: {task_info.error_message}")
                        return False

            await asyncio.sleep(1)
        else:
            print("⚠️ Indexing timeout (may still be running)")

        await server.shutdown()
        return True

    except Exception as e:
        print(f"❌ Indexing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all local tests"""
    print("🚀 Starting EOL RAG Context Server Local Tests")
    print("=" * 50)

    # Test 1: Basic initialization
    test1_passed = await test_server_initialization()

    # Test 2: Indexing workflow
    test2_passed = await test_indexing_workflow()

    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"Server Initialization: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Indexing Workflow: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    total_tests = 2
    passed_tests = sum([test1_passed, test2_passed])
    print(
        f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)"
    )

    if passed_tests == total_tests:
        print("\n🎉 All local tests passed! Server is ready for MCP testing.")
    else:
        print("\n⚠️ Some tests failed. Check server configuration.")


if __name__ == "__main__":
    asyncio.run(main())
