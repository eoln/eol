# Coverage Gap Analysis - EOL RAG Context

## Overall Coverage Status

- **Current Coverage**: 71.78%
- **Required Coverage**: 80%
- **Gap**: 8.22%

## Files with Critical Coverage Gaps

### 1. **server.py** (42.96% coverage - 162 lines missing)

**Uncovered Areas:**

- MCP resource handlers (lines 308-407)
  - `get_context_for_query` resource handler
  - `list_indexed_sources` resource handler
  - `get_statistics` resource handler
  - `get_knowledge_graph_stats` resource handler
- MCP tool implementations (lines 490-840)
  - Several tool handler internals
  - Error handling paths in tools
- Prompt setup methods (lines 1089-1223)
  - `structured_query_prompt`
  - `context_synthesis_prompt`
  - `knowledge_exploration_prompt`

**Recommendation**: Add integration tests for MCP handlers and prompts

### 2. **document_processor.py** (53.49% coverage - 221 lines missing)

**Uncovered Areas:**

- AST-based code chunking (lines 659-714)
  - Tree-sitter parsing for various languages
  - Node type detection for different languages
- XML processing methods (lines 857-902, 933-1066)
  - `_process_xml` method
  - `_extract_temporal_metadata`
  - `_chunk_xml_feed`, `_chunk_svg`, `_chunk_xml_config`
- PDF processing (lines 1070-1178)
  - `_process_pdf` method
  - PDF metadata extraction
  - `_chunk_pdf_content`
- DOCX processing (lines 1182-1214)
  - `_process_docx` method

**Recommendation**: Add tests for XML, PDF, and DOCX file processing

### 3. **async_task_manager.py** (62.43% coverage - 81 lines missing)

**Uncovered Areas:**

- Task cancellation logic (lines 340-394)
  - `cancel_task` method
  - Task cleanup procedures
- Error recovery paths (lines 416-422, 441-450)
- Task monitoring (lines 499-514, 527-565)
  - `_monitor_tasks` background task
  - Task expiration handling

**Recommendation**: Add tests for task lifecycle management

### 4. **parallel_indexer.py** (65.28% coverage - 65 lines missing)

**Uncovered Areas:**

- Parallel processing logic (lines 307-400)
  - `_process_batch` method
  - Worker pool management
- Error handling in batch operations (lines 413-420)
- Progress tracking (lines 436-441)

**Recommendation**: Add tests for parallel indexing scenarios

### 5. **redis_client.py** (65.71% coverage - 74 lines missing)

**Uncovered Areas:**

- Redis search module compatibility (lines 64-86)
  - Legacy FT.SEARCH imports (optional)
- Error handling paths (lines 477-502)
- Vector index management (lines 657-689)
  - Index creation error paths
  - Index update operations

**Recommendation**: Add tests with Redis mock for error scenarios

### 6. **file_watcher.py** (67.16% coverage - 86 lines missing)

**Uncovered Areas:**

- File system event handlers (lines 755-831)
  - `on_created`, `on_modified`, `on_deleted` handlers
- Pattern matching logic (lines 835-876)
- Cleanup procedures (lines 890-892)

**Recommendation**: Add tests for file system monitoring

## Files with Good Coverage (No Action Needed)

| File | Coverage | Status |
|------|----------|--------|
| config.py | 100% | ✅ Excellent |
| main.py | 100% | ✅ Excellent |
| embeddings.py | 92.89% | ✅ Good |
| batch_operations.py | 90.61% | ✅ Good |
| semantic_cache.py | 90.11% | ✅ Good |
| knowledge_graph.py | 89.38% | ✅ Good |
| indexer.py | 86.09% | ✅ Good |

## Priority Recommendations

### High Priority (Quick Wins)

1. **server.py**: Add tests for MCP resource handlers and prompts
2. **document_processor.py**: Add basic XML and PDF processing tests

### Medium Priority

1. **async_task_manager.py**: Add task lifecycle tests
2. **redis_client.py**: Add error scenario tests

### Low Priority

1. **parallel_indexer.py**: Add parallel processing tests
2. **file_watcher.py**: Add file monitoring tests

## Estimated Effort to Reach 80% Coverage

To increase coverage from 71.78% to 80%:

- Need to cover approximately 275 additional lines
- Focus on server.py and document_processor.py for maximum impact
- Estimated effort: 4-6 hours of test writing

## Test Implementation Strategy

### Quick Coverage Gains

1. Test MCP handlers in server.py (can mock dependencies)
2. Test document processing for common formats (MD, JSON, YAML)
3. Test error paths in async_task_manager.py

### Systematic Approach

1. Start with integration tests that cover multiple components
2. Add unit tests for specific uncovered methods
3. Focus on happy paths first, then error scenarios

## Current Test File Status

| Component | Source Lines | Test Lines | Coverage | Priority |
|-----------|--------------|------------|----------|----------|
| server.py | 327 | 343 | 42.96% | **HIGH** |
| document_processor.py | 536 | 1369 | 53.49% | **HIGH** |
| async_task_manager.py | 262 | 240 | 62.43% | MEDIUM |
| parallel_indexer.py | 213 | 319 | 65.28% | LOW |
| redis_client.py | 259 | 862 | 65.71% | MEDIUM |
| file_watcher.py | 305 | 625 | 67.16% | LOW |

## Specific Test Cases Needed

### For server.py (HIGH PRIORITY)

```python
# Add to test_server.py

async def test_mcp_resource_handlers(self):
    """Test MCP resource handlers."""
    # Test get_context_for_query
    # Test list_indexed_sources
    # Test get_statistics
    # Test get_knowledge_graph_stats

async def test_mcp_prompts(self):
    """Test MCP prompt generation."""
    # Test structured_query_prompt
    # Test context_synthesis_prompt
    # Test knowledge_exploration_prompt
```

### For document_processor.py (HIGH PRIORITY)

```python
# Add to test_document_processor.py

async def test_process_xml_formats(self):
    """Test XML file processing."""
    # Test RSS/Atom feeds
    # Test SVG files
    # Test generic XML

async def test_process_pdf(self):
    """Test PDF file processing."""
    # Test with pypdf
    # Test metadata extraction

async def test_process_docx(self):
    """Test DOCX file processing."""
    # Test with python-docx
```

### For async_task_manager.py (MEDIUM PRIORITY)

```python
# Add to test_async_task_manager.py

async def test_task_cancellation(self):
    """Test task cancellation scenarios."""
    # Test cancel_task method
    # Test cleanup procedures

async def test_task_monitoring(self):
    """Test background task monitoring."""
    # Test _monitor_tasks
    # Test task expiration
```

## Action Plan to Reach 80% Coverage

1. **Immediate Actions (2-3 hours)**
   - Add MCP handler tests to server.py (+~100 lines coverage)
   - Add XML/PDF tests to document_processor.py (+~100 lines coverage)
   - This should bring coverage to ~76-77%

2. **Secondary Actions (2-3 hours)**
   - Add task lifecycle tests to async_task_manager.py (+~40 lines)
   - Add error scenario tests to redis_client.py (+~30 lines)
   - This should bring coverage to ~79-80%

3. **Final Push if Needed (1-2 hours)**
   - Add parallel processing tests (+~30 lines)
   - Add file watcher event tests (+~40 lines)
   - This would bring coverage to >80%

## Commands to Verify Coverage

```bash
# Run coverage for specific files
pytest tests/unit/test_server.py --cov=src/eol/rag_context/server --cov-report=term-missing

# Run all tests with coverage
pytest tests/unit/ --cov=src/eol/rag_context --cov-report=html

# View HTML report
open htmlcov/index.html
```
