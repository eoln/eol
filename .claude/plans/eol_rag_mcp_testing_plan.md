# EOL RAG Context MCP Server - Comprehensive Testing Plan

## Overview

This plan provides systematic testing for all tools and resources in the `eol-rag-context` MCP server to ensure reliability, performance, and proper functionality of the non-blocking RAG indexing system.

## MCP Server Inventory

### Tools (12 total)

1. **start_indexing** - Start background indexing tasks
2. **get_indexing_status** - Check task progress
3. **list_indexing_tasks** - View active tasks
4. **cancel_indexing_task** - Stop running tasks
5. **cleanup_old_indexing_tasks** - Clean completed tasks
6. **search_context** - Search for relevant context
7. **query_knowledge_graph** - Query entity relationships
8. **optimize_context** - Optimize context for LLM consumption
9. **watch_directory** - Start file watching
10. **unwatch_directory** - Stop file watching
11. **clear_cache** - Clear all caches
12. **remove_source** - Remove indexed source

### Resources (3 total)

1. **list_indexed_sources** (context://sources) - List all indexed sources
2. **get_statistics** (context://stats) - Get server statistics
3. **get_knowledge_graph_stats** (context://knowledge-graph/stats) - Get graph stats

## Test Categories

### Category 1: Non-blocking Indexing Tools (Priority: HIGH)

**Tools**: start_indexing, get_indexing_status, list_indexing_tasks, cancel_indexing_task, cleanup_old_indexing_tasks

#### Test Scenarios

**1.1 Basic Non-blocking Workflow**

- Start indexing on a small test directory (5-10 files)
- Verify task returns immediately with task_id
- Monitor progress with get_indexing_status
- Confirm completion and cleanup

**1.2 Parameter Validation**

- Test start_indexing with invalid paths
- Test get_indexing_status with non-existent task_id
- Test list_indexing_tasks with invalid status filters
- Test cancel_indexing_task with invalid task_id

**1.3 Concurrent Task Management**

- Start multiple indexing tasks simultaneously
- List all active tasks
- Cancel specific tasks
- Verify task isolation

**1.4 Task Lifecycle Management**

- Test all task states: pending → running → completed
- Test failed task handling
- Test cleanup of old tasks
- Verify task persistence across server restarts

**1.5 Large Directory Indexing**

- Index directories with 100+, 1000+ files
- Monitor memory usage and performance
- Test cancellation of long-running tasks
- Verify progress reporting accuracy

### Category 2: Search and Retrieval Tools (Priority: HIGH)

**Tools**: search_context, query_knowledge_graph, optimize_context

#### Test Scenarios

**2.1 Search Functionality**

- Index test documents with known content
- Search for specific terms and phrases
- Verify relevance scoring and ranking
- Test semantic similarity matching

**2.2 Knowledge Graph Queries**

- Verify entity extraction from documents
- Test relationship queries
- Validate graph statistics accuracy
- Test query performance with large graphs

**2.3 Context Optimization**

- Test context window management
- Verify LLM-optimized formatting
- Test compression strategies
- Validate context relevance filtering

### Category 3: File Management Tools (Priority: MEDIUM)

**Tools**: watch_directory, unwatch_directory, remove_source

#### Test Scenarios

**3.1 File Watching**

- Start watching a directory
- Add, modify, delete files
- Verify automatic reindexing
- Test watch performance with many files

**3.2 Source Management**

- Index multiple sources
- Remove specific sources
- Verify data cleanup
- Test source isolation

### Category 4: System Management Tools (Priority: MEDIUM)

**Tools**: clear_cache

#### Test Scenarios

**4.1 Cache Management**

- Build up cache data through searches
- Clear caches and verify cleanup
- Test cache rebuild performance
- Verify cache hit/miss statistics

### Category 5: MCP Resources (Priority: HIGH)

**Resources**: context://sources, context://stats, context://knowledge-graph/stats

#### Test Scenarios

**5.1 Resource Accessibility**

- Test all resources return valid JSON
- Verify resource data accuracy
- Test resource performance under load
- Validate resource schema consistency

**5.2 Statistics Validation**

- Compare resource stats with actual data
- Verify statistics update in real-time
- Test statistics across different server states

## Test Data Setup

### Small Test Dataset (for basic functionality)

```
test-docs/
├── README.md (basic markdown)
├── code.py (Python code)
├── config.json (JSON data)
├── notes.txt (plain text)
└── subdir/
    ├── more-code.js (JavaScript)
    └── documentation.md (markdown)
```

### Medium Test Dataset (for performance testing)

- 50-100 mixed file types
- Various sizes (1KB to 1MB)
- Multiple directories (3-5 levels deep)
- Include binary files (should be skipped)

### Large Test Dataset (for stress testing)

- 1000+ files
- Realistic codebase structure
- Mix of programming languages
- Documentation files

## Test Execution Plan

### Phase 1: Smoke Tests (Essential functionality)

1. Server startup and connection
2. Basic tool parameter validation
3. Resource accessibility
4. Simple indexing workflow

### Phase 2: Core Functionality Tests

1. Complete non-blocking indexing workflow
2. Search and retrieval accuracy
3. Statistics and monitoring
4. Error handling and edge cases

### Phase 3: Performance and Stress Tests

1. Large directory indexing
2. Concurrent operations
3. Memory usage monitoring
4. Long-running task management

### Phase 4: Integration Tests

1. End-to-end RAG pipeline
2. File watching integration
3. Cache behavior validation
4. Knowledge graph construction

## Success Criteria

### Functional Requirements

- [ ] All tools execute without errors
- [ ] Non-blocking operations don't block client
- [ ] Search returns relevant results
- [ ] Statistics accurately reflect system state
- [ ] File watching triggers reindexing
- [ ] Task cancellation works reliably

### Performance Requirements

- [ ] Indexing rate > 10 documents/second
- [ ] Search latency < 100ms for 10k documents
- [ ] Memory usage remains stable
- [ ] Task overhead < 10% of total processing time

### Reliability Requirements

- [ ] No memory leaks during extended operation
- [ ] Graceful error handling for all edge cases
- [ ] Server recovery after component failures
- [ ] Data consistency across operations

## Test Environment Setup

### Prerequisites

- Redis server running (version 8.2+)
- Python virtual environment activated
- Test data directories prepared
- MCP server configured and running

### Monitoring Tools

- Memory usage monitoring
- Redis monitoring (RedisInsight)
- Performance profiling tools
- Log aggregation for error tracking

## Automated Test Execution

### Test Script Structure

```python
# Automated test runner
async def run_test_suite():
    # Phase 1: Smoke tests
    await test_server_connectivity()
    await test_basic_tool_validation()

    # Phase 2: Core functionality
    await test_nonblocking_workflow()
    await test_search_accuracy()

    # Phase 3: Performance tests
    await test_large_directory_indexing()
    await test_concurrent_operations()

    # Phase 4: Integration tests
    await test_end_to_end_rag_pipeline()
```

## Risk Mitigation

### High-Risk Areas

1. **Concurrent task management** - Risk of race conditions
2. **Memory usage** - Risk of memory leaks with large datasets
3. **Redis integration** - Risk of data corruption
4. **File watching** - Risk of missing file changes

### Mitigation Strategies

- Comprehensive error logging
- Resource usage monitoring
- Backup and recovery procedures
- Incremental testing approach

## Test Reporting

### Metrics to Track

- Test execution time
- Success/failure rates
- Performance benchmarks
- Resource utilization
- Error frequencies

### Report Format

- Executive summary
- Detailed test results by category
- Performance analysis
- Issue recommendations
- Next steps

---

*This testing plan ensures comprehensive validation of the eol-rag-context MCP server functionality, performance, and reliability.*
