# EOL CLI

**Status**: 📋 Planned
**Target Release**: Q2 2025

## Overview

EOL CLI will provide a powerful command-line interface for managing RAG pipelines, deployments, and monitoring. It will offer both interactive and scriptable commands for all EOL operations.

## Planned Features

### Pipeline Management

- Create and manage RAG pipelines
- Index documents from CLI
- Search and query interfaces
- Pipeline monitoring

### Deployment Tools

- Docker deployment helpers
- Kubernetes manifests generation
- Configuration management
- Health checks

### Development Tools

- Project scaffolding
- Testing utilities
- Performance benchmarking
- Debug tools

### Monitoring

- Real-time metrics
- Log aggregation
- Performance dashboards
- Alert configuration

## Planned Commands

```bash
# Pipeline management
eol pipeline create my-rag-pipeline
eol pipeline index ./documents
eol pipeline search "query text"

# Deployment
eol deploy docker
eol deploy k8s --namespace eol

# Monitoring
eol status
eol metrics
eol logs --follow

# Development
eol new project my-app
eol test
eol benchmark
```

## Design Goals

- **User-Friendly**: Intuitive commands with helpful output
- **Scriptable**: JSON output for automation
- **Extensible**: Plugin system for custom commands
- **Fast**: Minimal overhead and quick response

## Contributing

Want to help shape EOL CLI?

1. Share your ideas in [Discussions](https://github.com/eoln/eol/discussions)
2. Review the [CLI Design Document](https://github.com/eoln/eol/wiki/CLI-Design) (coming soon)
3. Check [planned features](https://github.com/eoln/eol/issues?q=is%3Aopen+is%3Aissue+label%3Aeol-cli)

---

This package is in the planning phase. Your input is welcome!
