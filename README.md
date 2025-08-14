# EOL Framework

EOL is a comprehensive AI framework for building intelligent, context-aware applications.

## Overview

This is a monorepo containing multiple packages:

- **eol-rag-context**: RAG-based context management MCP server
- **eol-core**: Core framework utilities (planned)
- **eol-cli**: Command-line interface (planned)

## Development

This project uses UV for dependency management and workspace management.

```bash
# Install dependencies
uv sync --all-packages

# Run tests
./test_all.sh
```

## License

GPL-3.0
