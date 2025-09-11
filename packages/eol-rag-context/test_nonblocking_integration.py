#!/usr/bin/env python3
"""Integration tests for non-blocking MCP server."""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eol.rag_context.server import EOLRAGContextServer
from eol.rag_context.config import RAGConfig


async def test_nonblocking_tools():
    """Test non-blocking indexing functionality directly."""
    print("🧪 Testing Non-Blocking Indexing")
    print("=" * 60)
    
    try:
        # Initialize server
        print("\n📡 Initializing server...")
        config = RAGConfig()
        server = EOLRAGContextServer(config)
        await server.initialize()
        print("✅ Server initialized successfully")
        
        # Test directory path
        test_data_path = str(Path(__file__).parent / "test_integration_data")
        print(f"📁 Test directory: {test_data_path}")
        
        # Test 1: Test the new non-blocking index_directory method
        print("\n🚀 Test 1: index_directory (should return immediately with task ID)")
        start_time = time.time()
        
        result = await server.index_directory(
            path=test_data_path,
            description="Integration test indexing"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Duration: {duration:.3f} seconds")
        print(f"📄 Result: {result}")
        
        if duration > 1.0:  # Should be nearly instantaneous
            print("⚠️  WARNING: index_directory took more than 1 second - may be blocking!")
        else:
            print("✅ index_directory returned immediately (non-blocking)")
        
        # Check if task_id is returned (new non-blocking behavior)
        task_id = result.get("task_id")
        if task_id:
            print(f"✅ Task ID returned: {task_id} (non-blocking mode confirmed)")
            
            # Test 2: Check if task manager has our task
            print(f"\n📊 Test 2: Checking task manager for task {task_id}")
            
            status_checks = 0
            max_status_checks = 10
            
            while status_checks < max_status_checks:
                # Check task status through task manager
                task_info = await server.task_manager.get_task_status(task_id)
                print(f"📋 Status check {status_checks + 1}: {task_info}")
                
                if task_info and task_info.status.value in ["completed", "failed", "cancelled"]:
                    print(f"✅ Task reached final status: {task_info.status.value}")
                    break
                
                status_checks += 1
                if status_checks < max_status_checks:
                    await asyncio.sleep(1)  # Wait 1 second between checks
            
            # Test 3: List all tasks
            print("\n📋 Test 3: list_indexing_tasks")
            all_tasks = await server.task_manager.list_tasks()
            print(f"📝 All tasks: {all_tasks}")
            
        else:
            print("⚠️  No task_id returned - may be running in blocking mode")
        
        print("\n🎉 Non-blocking indexing tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"🔍 Traceback:")
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            # Cleanup server resources
            if hasattr(server, 'redis_store') and server.redis_store:
                await server.redis_store.close()
            print("\n🧹 Server cleanup completed")
        except Exception as e:
            print(f"⚠️  Warning: Cleanup error: {e}")


async def test_system_responsiveness():
    """Test that the system remains responsive during indexing."""
    print("\n⚡ Testing System Responsiveness During Indexing")
    print("=" * 60)
    
    try:
        config = RAGConfig()
        server = EOLRAGContextServer(config)
        await server.initialize()
        
        test_data_path = str(Path(__file__).parent / "test_integration_data")
        
        # Start indexing
        print("🚀 Starting indexing task...")
        result = await server.index_directory(
            path=test_data_path,
            description="Responsiveness test"
        )
        task_id = result.get("task_id")
        
        if task_id:
            # Immediately perform other operations to test responsiveness
            print("🏃 Performing concurrent operations to test responsiveness...")
            
            operations = []
            
            # Test multiple concurrent status checks
            for i in range(5):
                operations.append(server.task_manager.get_task_status(task_id))
            
            # Test listing tasks
            operations.append(server.task_manager.list_tasks())
            
            # Execute all operations concurrently
            start_time = time.time()
            results = await asyncio.gather(*operations, return_exceptions=True)
            end_time = time.time()
            
            print(f"⏱️  All {len(operations)} operations completed in {end_time - start_time:.3f} seconds")
            
            # Check if any operation failed
            failed_count = sum(1 for result in results if isinstance(result, Exception))
            
            if failed_count == 0:
                print("✅ All concurrent operations succeeded - system is responsive!")
            else:
                print(f"⚠️  {failed_count}/{len(operations)} operations failed during concurrent execution")
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"   Operation {i+1} failed: {result}")
        else:
            print("⚠️  No task_id returned - cannot test responsiveness with task operations")
            
        # Cleanup
        if hasattr(server, 'redis_store') and server.redis_store:
            await server.redis_store.close()
        
    except Exception as e:
        print(f"❌ Responsiveness test failed: {e}")
        traceback.print_exc()


async def main():
    """Main test function."""
    print("🚀 Starting Non-Blocking MCP Integration Tests")
    print("=" * 80)
    
    # Run tool tests
    await test_nonblocking_tools()
    
    # Run responsiveness tests
    await test_system_responsiveness()
    
    print("\n🏁 All integration tests completed!")


if __name__ == "__main__":
    asyncio.run(main())