#!/bin/bash
# Project-specific MCP launcher
# Copy this to your project root and customize as needed

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create project-specific directories if they don't exist
mkdir -p "$PROJECT_ROOT/.rag-data"
mkdir -p "$PROJECT_ROOT/.rag-index"

# Export environment variables
export EOL_RAG_DATA_DIR="$PROJECT_ROOT/.rag-data"
export EOL_RAG_INDEX_DIR="$PROJECT_ROOT/.rag-index"

# Launch the MCP server
exec "$HOME/.local/bin/eol-rag-mcp" "$@"
