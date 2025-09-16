# EOL RAG Context - Intelligent Memory for Claude Code

## What is This?

Hello! 👋 This is an MCP (Model Context Protocol) server that gives Claude Code intelligent memory and document understanding capabilities. If you're new to RAG (Retrieval-Augmented Generation), think of it as giving AI the ability to remember and search through your documents intelligently - like having a super-smart assistant who never forgets what you've shown them.

### What's RAG?

RAG combines the power of AI with your specific documents and code. Instead of relying only on general knowledge, Claude can search through YOUR files and give you answers based on YOUR content.

### What's MCP?

MCP (Model Context Protocol) is how Claude Code communicates with external tools. This server acts as a bridge, allowing Claude to index your documents and search through them when answering your questions.

## Quick Start

### Prerequisites

- Python 3.13+
- Redis 8.2+ (with native Vector Sets support)
- uv package manager

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/eol.git
cd eol/packages/eol-rag-context
uv venv
uv sync
```

### Setting up with Claude Code

1. Edit your Claude Code configuration:

```bash
# Open Claude Code config
claude config edit

# Or directly edit:
~/.config/claude-code/config.json
```

2. Add the EOL RAG Context server:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "uv",
      "args": ["run", "eol-rag-mcp"],
      "cwd": "/path/to/eol/packages/eol-rag-context"
    }
  }
}
```

3. Test it's working:

```bash
claude "What tools are available from the eol-rag-context server?"
```

## What Can It Do?

Once set up, you can ask Claude Code things like:

- "Index all Python files in my project"
- "Search for authentication implementations in the codebase"
- "What does the document processor do?"
- "Find all references to Redis configuration"
- "Build a knowledge graph of my documentation"

### Available Tools

The server provides 12 MCP tools:

- **search_context** - Search through indexed documents
- **start_indexing** - Begin indexing files or directories
- **get_indexing_status** - Check indexing progress
- **get_indexing_stats** - View indexing statistics
- **list_indexed_sources** - See what's been indexed
- **remove_indexed_source** - Remove indexed content
- **optimize_index** - Improve search performance
- **get_cache_stats** - View semantic cache performance
- **clear_cache** - Clear the semantic cache
- **query_knowledge_graph** - Query entity relationships
- **visualize_knowledge_graph** - Generate graph visualizations
- **watch_directory** - Auto-index file changes

## Basic Usage Examples

```bash
# Index your project
claude "Please index all Python files in the current directory"

# Search for specific concepts
claude "Search for all database connection code in the indexed files"

# Get insights
claude "What are the main components of this system based on the indexed code?"

# Monitor changes
claude "Watch the src directory for changes and auto-index them"
```

## Configuration

Configuration is simple and flexible. See [configuration.md](configuration.md) for details.

Default configuration location: `~/.config/eol-rag/config.yaml`

## Examples

Check out the [examples/](examples/) directory for:

- `quick_start.py` - Basic indexing and search
- `code_assistant.py` - Code analysis helper
- `rag_cli.py` - Command-line RAG interface

## Development

```bash
# Run tests
uv run pytest

# Run with verbose logging
uv run eol-rag-mcp --verbose
```

## License

GPL-3.0 - See LICENSE file for details.

## Need Help?

- Check the [examples/](examples/) for working code
- Configuration options in [configuration.md](configuration.md)
- Open an issue on GitHub for bugs or questions

---

*Built with ❤️ for the Claude Code community*
