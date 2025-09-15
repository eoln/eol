# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **MAJOR**: Native Redis 8.2+ Vector Sets support with VADD/VSIM commands
- Q8 quantization for memory-efficient vector storage
- SVS-VAMANA algorithm for high-performance similarity search
- Hierarchical Vector Set organization (concepts, sections, chunks)
- Comprehensive Vector Set testing suite (397 unit + 52 integration tests)
- Binary data handling improvements for Vector Set operations
- Comprehensive CI/CD dependencies in pyproject.toml for improved caching
- Python matrix testing (3.11, 3.12, 3.13) with isolated Redis instances
- Wheel cache optimization with dependency hash-based caching
- Unified GitHub workflow (ci-cd.yml) consolidating all CI processes

### Changed

- **BREAKING**: Migrated from Redis FT.SEARCH to native Redis 8.2+ Vector Sets
- **BREAKING**: Now requires Redis 8.2+ instead of Redis Stack with RediSearch module
- **BREAKING**: Updated IndexConfig to use Vector Set parameters (SVS-VAMANA, Q8 quantization)
- **BREAKING**: Replaced all FT.CREATE/FT.SEARCH operations with VADD/VSIM Vector Set commands
- Updated semantic cache to use Vector Sets with proper UTF-8/binary data handling
- Migrated all test infrastructure to mock Vector Set commands instead of FT.SEARCH
- **BREAKING**: Migrated from pip to uv as primary package manager for 10x faster installs
- Updated all documentation to reflect Redis 8.2+ Vector Sets and uv installation
- Migrated Dockerfile.test from pip to uv for faster container builds
- Consolidated GitHub workflows into single unified ci-cd.yml

### Removed

- **BREAKING**: Removed dependency on Redis Stack and RediSearch module
- **BREAKING**: Removed all FT.SEARCH/FT.CREATE legacy code and configurations

### Fixed

- Prevented redundant wheel cache artifact uploads
- Improved wheel cache installation for Redis dependencies
- Addressed quality check and security audit failures in CI/CD workflow
- Test isolation issues with sys.modules modification at import time

### Performance

- Removed redundant venv caching to eliminate post-job overhead
- Optimized CI/CD build times with dependency caching
- Improved test execution speed with parallel job matrix

## [0.1.0] - 2024-08-10

### Added

- Initial release of EOL RAG Context MCP Server
- Hierarchical indexing with 3-level structure (concepts → sections → chunks)
- Knowledge graph with automatic entity extraction
- Real-time file watching with automatic reindexing
- Semantic caching with 31% hit rate optimization
- Multi-format support (Markdown, PDF, DOCX, JSON/YAML, source code)
- MCP Protocol implementation for Model Context Protocol
- Redis Stack 8.0+ integration for vector storage
- Comprehensive test suite with 80% coverage target
