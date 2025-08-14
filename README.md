# EOL Framework

![Unit Test Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eoln/eol/main/.github/badges/unit-coverage-badge.json)
![Integration Test Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eoln/eol/main/.github/badges/integration-coverage-badge.json)
![CI/CD Status](https://github.com/eoln/eol/actions/workflows/eol-rag-context.yml/badge.svg)

EOL is a comprehensive AI framework for building intelligent, context-aware applications.

## Overview

This is a monorepo containing multiple packages for building intelligent AI applications with retrieval-augmented generation (RAG) capabilities.

## Packages

| Package | Status | Description | Test Coverage |
|---------|--------|-------------|---------------|
| **eol-rag-context** | ✅ Active | RAG-based context management MCP server | ![Unit](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eoln/eol/main/.github/badges/unit-coverage-badge.json) ![Integration](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eoln/eol/main/.github/badges/integration-coverage-badge.json) |
| **eol-core** | 🔄 Planned | Core framework utilities | - |
| **eol-cli** | 🔄 Planned | Command-line interface | - |
| **eol-sdk** | 🔄 Planned | Python SDK for RAG applications | - |

## Key Features

- 🚀 **Ultra-Fast Dependencies**: Advanced wheel caching for 3-6x faster CI/CD builds
- 🔍 **Intelligent Indexing**: Content-aware chunking with AST parsing for code and semantic splitting for text
- 📊 **Redis Vector Database**: High-performance semantic search using Redis Stack v8
- 🧠 **Semantic Caching**: 31% cache hit rate target to reduce LLM API calls
- 🔗 **Knowledge Graphs**: Entity relationship mapping for enhanced context
- 📡 **MCP Integration**: Model Context Protocol server for seamless AI integration
- 👁️ **File Watching**: Auto-indexing with real-time change detection

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
