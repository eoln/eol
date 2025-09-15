# EOL RAG Context - Package Context

## Overview

MCP server providing intelligent RAG (Retrieval-Augmented Generation) capabilities to Claude Code.

## Key Facts

- **Python**: 3.13+ required
- **Redis**: 8.2+ with native Vector Sets
- **Package Manager**: uv only
- **License**: GPL-3.0

## Available MCP Tools (12)

1. `search_context` - Vector search through indexed documents
2. `start_indexing` - Index files or directories
3. `get_indexing_status` - Check indexing progress
4. `get_indexing_stats` - View statistics
5. `list_indexed_sources` - List indexed content
6. `remove_indexed_source` - Remove from index
7. `optimize_index` - Improve performance
8. `get_cache_stats` - Cache metrics
9. `clear_cache` - Clear semantic cache
10. `query_knowledge_graph` - Query relationships
11. `visualize_knowledge_graph` - Generate visualizations
12. `watch_directory` - Auto-index changes

## Testing

```bash
# Run all tests
./test_all.sh

# Unit tests only
uv run pytest tests/test_*.py

# Integration tests
uv run pytest tests/integration/
```

## Configuration

User config: `~/.config/eol-rag/config.yaml`
See: [configuration.md](../configuration.md)

## Examples

- `examples/quick_start.py` - Basic usage
- `examples/code_assistant.py` - Interactive assistant
- `examples/rag_cli.py` - CLI interface
