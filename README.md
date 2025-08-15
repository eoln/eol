# EOL Framework

![Unit Test Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eoln/eol/main/.github/badges/unit-coverage-badge.json)
![Integration Test Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eoln/eol/main/.github/badges/integration-coverage-badge.json)
![CI/CD Status](https://github.com/eoln/eol/actions/workflows/eol-rag-context.yml/badge.svg)

## 🚀 Performance Metrics

![Indexing Speed](https://img.shields.io/badge/Indexing-15.3_docs%2Fs-success)
![Search Latency](https://img.shields.io/badge/Search-87ms-success)
![Cache Hit Rate](https://img.shields.io/badge/Cache_Hit-34.2%25-success)

<details>
<summary>📊 View Detailed Performance Benchmarks</summary>

| Component | Metric | Current | Target | Status |
|-----------|--------|---------|--------|--------|
| **Document Processing** | | | | |
| Indexing Speed | Files/sec | 15.3 | >10 | ✅ |
| Chunk Processing | Chunks/sec | 48.2 | >40 | ✅ |
| **Vector Search** | | | | |
| Query Latency (P50) | Milliseconds | 87 | <100 | ✅ |
| Query Latency (P95) | Milliseconds | 142 | <200 | ✅ |
| Searches/sec | Operations | 11.5 | >10 | ✅ |
| **Semantic Cache** | | | | |
| Hit Rate | Percentage | 34.2% | >31% | ✅ |
| Read Latency | Milliseconds | 12 | <20 | ✅ |
| Write Latency | Milliseconds | 45 | <250 | ✅ |

*Performance metrics are automatically updated by CI/CD pipeline. Last update: see [workflow runs](https://github.com/eoln/eol/actions/workflows/ci-cd.yml)*

</details>

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
