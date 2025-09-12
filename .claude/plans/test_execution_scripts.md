# Test Execution Scripts for EOL RAG Context MCP Server

## Automated Test Runner Implementation

### Core Test Framework

```python
#!/usr/bin/env python3
"""
Comprehensive test suite for EOL RAG Context MCP Server
Run with: python test_eol_rag_mcp.py
"""

import asyncio
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Test configuration
TEST_CONFIG = {
    "small_dataset_path": "/tmp/test-docs-small",
    "medium_dataset_path": "/tmp/test-docs-medium", 
    "large_dataset_path": "/tmp/test-docs-large",
    "timeout_seconds": 300,
    "max_concurrent_tasks": 5
}

class MCPTestClient:
    """MCP client for testing tools and resources"""
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call an MCP tool with parameters"""
        # Implementation would use actual MCP client
        pass
    
    async def get_resource(self, uri: str) -> Dict[str, Any]:
        """Get an MCP resource"""
        # Implementation would use actual MCP client
        pass

class TestResult:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error_message = None
        self.execution_time = 0
        self.metrics = {}

class EOLRAGMCPTestSuite:
    def __init__(self):
        self.client = MCPTestClient()
        self.results: List[TestResult] = []
        
    async def run_all_tests(self):
        """Execute complete test suite"""
        print("🚀 Starting EOL RAG Context MCP Server Test Suite")
        
        # Phase 1: Smoke Tests
        await self.run_smoke_tests()
        
        # Phase 2: Core Functionality
        await self.run_core_functionality_tests()
        
        # Phase 3: Performance Tests
        await self.run_performance_tests()
        
        # Phase 4: Integration Tests
        await self.run_integration_tests()
        
        # Generate report
        self.generate_test_report()
    
    # ==================== SMOKE TESTS ====================
    
    async def run_smoke_tests(self):
        """Basic connectivity and validation tests"""
        print("\n📋 Phase 1: Smoke Tests")
        
        await self.test_server_connectivity()
        await self.test_resource_accessibility()
        await self.test_basic_tool_validation()
    
    async def test_server_connectivity(self):
        """Test basic MCP server connection"""
        result = TestResult("server_connectivity")
        start_time = time.time()
        
        try:
            # Test getting server statistics
            stats = await self.client.get_resource("context://stats")
            assert "indexer" in stats
            assert "cache" in stats
            result.passed = True
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def test_resource_accessibility(self):
        """Test all MCP resources are accessible"""
        resources = [
            "context://sources",
            "context://stats", 
            "context://knowledge-graph/stats"
        ]
        
        for uri in resources:
            result = TestResult(f"resource_{uri.split('/')[-1]}")
            start_time = time.time()
            
            try:
                data = await self.client.get_resource(uri)
                assert data is not None
                result.passed = True
            except Exception as e:
                result.error_message = str(e)
            
            result.execution_time = time.time() - start_time
            self.results.append(result)
    
    async def test_basic_tool_validation(self):
        """Test tool parameter validation"""
        test_cases = [
            # Valid calls
            ("cleanup_old_indexing_tasks", {}, True),
            ("list_indexing_tasks", {}, True),
            
            # Invalid calls  
            ("get_indexing_status", {"task_id": "nonexistent"}, False),
            ("cancel_indexing_task", {"task_id": "invalid"}, False),
        ]
        
        for tool_name, params, should_succeed in test_cases:
            result = TestResult(f"validation_{tool_name}")
            start_time = time.time()
            
            try:
                response = await self.client.call_tool(tool_name, **params)
                if should_succeed:
                    result.passed = "error" not in response
                else:
                    result.passed = "error" in response
            except Exception as e:
                result.passed = not should_succeed
                result.error_message = str(e)
            
            result.execution_time = time.time() - start_time
            self.results.append(result)
    
    # ==================== CORE FUNCTIONALITY TESTS ====================
    
    async def run_core_functionality_tests(self):
        """Test main functionality of all tools"""
        print("\n🔧 Phase 2: Core Functionality Tests")
        
        await self.test_nonblocking_indexing_workflow()
        await self.test_search_and_retrieval()
        await self.test_file_management()
        await self.test_cache_management()
    
    async def test_nonblocking_indexing_workflow(self):
        """Test complete non-blocking indexing workflow"""
        result = TestResult("nonblocking_workflow")
        start_time = time.time()
        
        try:
            # 1. Start indexing
            response = await self.client.call_tool(
                "start_indexing", 
                path=TEST_CONFIG["small_dataset_path"]
            )
            task_id = response.get("task_id")
            assert task_id is not None
            
            # 2. Monitor progress
            max_attempts = 30
            for attempt in range(max_attempts):
                status = await self.client.call_tool(
                    "get_indexing_status",
                    task_id=task_id
                )
                
                if status.get("status") == "completed":
                    break
                elif status.get("status") == "failed":
                    raise Exception(f"Indexing failed: {status.get('error')}")
                
                await asyncio.sleep(1)
            else:
                raise Exception("Indexing did not complete within timeout")
            
            # 3. Verify results
            sources = await self.client.get_resource("context://sources")
            assert len(sources) > 0
            
            # 4. Cleanup
            await self.client.call_tool("cleanup_old_indexing_tasks")
            
            result.passed = True
            result.metrics = {
                "task_id": task_id,
                "completion_time": time.time() - start_time,
                "files_indexed": status.get("total_files", 0)
            }
            
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    async def test_search_and_retrieval(self):
        """Test search functionality"""
        result = TestResult("search_and_retrieval")
        start_time = time.time()
        
        try:
            # Ensure we have indexed content
            sources = await self.client.get_resource("context://sources")
            if len(sources) == 0:
                # Index test data first
                await self.client.call_tool(
                    "start_indexing",
                    path=TEST_CONFIG["small_dataset_path"]
                )
                await asyncio.sleep(5)  # Wait for indexing
            
            # Test search
            search_results = await self.client.call_tool(
                "search_context",
                query="test content",
                max_results=5
            )
            
            assert len(search_results) > 0
            assert all("score" in result for result in search_results)
            
            result.passed = True
            result.metrics = {
                "results_count": len(search_results),
                "avg_score": sum(r["score"] for r in search_results) / len(search_results)
            }
            
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    # ==================== PERFORMANCE TESTS ====================
    
    async def run_performance_tests(self):
        """Test performance with larger datasets"""
        print("\n⚡ Phase 3: Performance Tests")
        
        await self.test_concurrent_indexing()
        await self.test_large_directory_performance()
        await self.test_memory_usage()
    
    async def test_concurrent_indexing(self):
        """Test multiple concurrent indexing tasks"""
        result = TestResult("concurrent_indexing")
        start_time = time.time()
        
        try:
            # Start multiple indexing tasks
            tasks = []
            for i in range(TEST_CONFIG["max_concurrent_tasks"]):
                task_response = await self.client.call_tool(
                    "start_indexing",
                    path=f"{TEST_CONFIG['medium_dataset_path']}_{i}"
                )
                tasks.append(task_response["task_id"])
            
            # Monitor all tasks
            completed_tasks = 0
            max_wait_time = 60
            start_wait = time.time()
            
            while completed_tasks < len(tasks) and (time.time() - start_wait) < max_wait_time:
                for task_id in tasks:
                    status = await self.client.call_tool(
                        "get_indexing_status",
                        task_id=task_id
                    )
                    if status.get("status") in ["completed", "failed"]:
                        completed_tasks += 1
                
                await asyncio.sleep(1)
            
            result.passed = completed_tasks == len(tasks)
            result.metrics = {
                "concurrent_tasks": len(tasks),
                "completed_tasks": completed_tasks,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    # ==================== INTEGRATION TESTS ====================
    
    async def run_integration_tests(self):
        """Test end-to-end integration scenarios"""
        print("\n🔗 Phase 4: Integration Tests")
        
        await self.test_end_to_end_rag_pipeline()
        await self.test_file_watching_integration()
    
    async def test_end_to_end_rag_pipeline(self):
        """Test complete RAG pipeline from indexing to search"""
        result = TestResult("end_to_end_rag")
        start_time = time.time()
        
        try:
            # 1. Clear existing data
            await self.client.call_tool("clear_cache")
            
            # 2. Index documents
            index_response = await self.client.call_tool(
                "start_indexing",
                path=TEST_CONFIG["small_dataset_path"]
            )
            
            # Wait for completion
            task_id = index_response["task_id"]
            await self.wait_for_task_completion(task_id)
            
            # 3. Verify indexing statistics
            stats = await self.client.get_resource("context://stats")
            assert stats["indexer"]["documents_indexed"] > 0
            
            # 4. Perform searches
            search_results = await self.client.call_tool(
                "search_context",
                query="function definition",
                max_results=3
            )
            
            # 5. Optimize context
            optimized = await self.client.call_tool(
                "optimize_context",
                context_items=search_results[:2],
                max_tokens=1000
            )
            
            # 6. Query knowledge graph
            graph_stats = await self.client.get_resource("context://knowledge-graph/stats")
            
            result.passed = True
            result.metrics = {
                "documents_indexed": stats["indexer"]["documents_indexed"],
                "search_results": len(search_results),
                "entities_found": graph_stats.get("entity_count", 0)
            }
            
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        self.results.append(result)
    
    # ==================== UTILITY METHODS ====================
    
    async def wait_for_task_completion(self, task_id: str, timeout: int = 60) -> bool:
        """Wait for a task to complete"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            status = await self.client.call_tool(
                "get_indexing_status",
                task_id=task_id
            )
            
            if status.get("status") == "completed":
                return True
            elif status.get("status") == "failed":
                raise Exception(f"Task failed: {status.get('error')}")
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Test Report")
        print(f"=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.test_name}: {result.error_message}")
        
        # Performance metrics
        print(f"\n⚡ Performance Summary:")
        total_time = sum(r.execution_time for r in self.results)
        print(f"Total Execution Time: {total_time:.2f}s")
        
        # Save detailed report
        report_data = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests/total_tests,
                "total_time": total_time
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "metrics": r.metrics
                }
                for r in self.results
            ]
        }
        
        with open("test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: test_report.json")

# ==================== TEST EXECUTION ====================

async def main():
    """Main test execution"""
    # Setup test data
    await setup_test_data()
    
    # Run test suite
    test_suite = EOLRAGMCPTestSuite()
    await test_suite.run_all_tests()

async def setup_test_data():
    """Create test data directories and files"""
    test_dirs = [
        TEST_CONFIG["small_dataset_path"],
        TEST_CONFIG["medium_dataset_path"],
        TEST_CONFIG["large_dataset_path"]
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        # Create sample files
        (Path(test_dir) / "README.md").write_text("# Test Documentation\nThis is test content for indexing.")
        (Path(test_dir) / "code.py").write_text("def test_function():\n    return 'hello world'")
        (Path(test_dir) / "config.json").write_text('{"setting": "value", "number": 42}')

if __name__ == "__main__":
    asyncio.run(main())
```

## Quick Test Commands

### Manual Testing Commands
```bash
# Test basic connectivity
/mcp
mcp__eol-rag-context__cleanup_old_indexing_tasks

# Test resource access
ReadMcpResourceTool server=eol-rag-context uri=context://stats

# Test indexing workflow
mcp__eol-rag-context__start_indexing path="/path/to/test/dir"
mcp__eol-rag-context__get_indexing_status task_id="<task_id>"
mcp__eol-rag-context__list_indexing_tasks

# Test search functionality
mcp__eol-rag-context__search_context query="test content" max_results=5
```

### Performance Testing Script
```bash
#!/bin/bash
# performance_test.sh

echo "🚀 Running EOL RAG Context Performance Tests"

# Test concurrent indexing
for i in {1..5}; do
    echo "Starting indexing task $i"
    # Start indexing in background
    # mcp__eol-rag-context__start_indexing path="/test/dir$i" &
done

# Monitor progress
echo "Monitoring task progress..."
# List active tasks periodically
# while [ $(mcp__eol-rag-context__list_indexing_tasks | jq '.total_tasks') -gt 0 ]; do
#     sleep 2
#     echo "Tasks still running..."
# done

echo "✅ Performance test completed"
```

## Test Data Generation

### Small Dataset Generator
```python
def create_small_test_dataset():
    base_path = Path("/tmp/test-docs-small")
    base_path.mkdir(exist_ok=True)
    
    files = {
        "README.md": "# Test Project\nThis is a test project for indexing.",
        "main.py": "def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()",
        "config.json": '{"database": "sqlite", "debug": true}',
        "docs/api.md": "# API Documentation\n## Endpoints\n- GET /users\n- POST /users",
        "src/utils.py": "def helper_function(data):\n    return data.upper()"
    }
    
    for file_path, content in files.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
```

This comprehensive testing plan covers all aspects of the eol-rag-context MCP server functionality.