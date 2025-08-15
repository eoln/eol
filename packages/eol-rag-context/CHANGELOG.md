# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI/CD dependencies in pyproject.toml for improved caching
- Python matrix testing (3.11, 3.12, 3.13) with isolated Redis instances
- Wheel cache optimization with dependency hash-based caching
- Unified GitHub workflow (ci-cd.yml) consolidating all CI processes

### Changed
- **BREAKING**: Migrated from pip to uv as primary package manager for 10x faster installs
- Updated all documentation to reflect uv as the recommended installation method
- Migrated Dockerfile.test from pip to uv for faster container builds
- Consolidated GitHub workflows into single unified ci-cd.yml

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
