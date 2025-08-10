# EOL - AI Framework for Building Modern LLM Applications

EOL is a comprehensive AI framework designed to build modern applications leveraging Large Language Models (LLMs). It provides a unique two-phase development model that enables rapid prototyping with natural language specifications and seamless transition to production-ready implementations.

## 🚀 Key Features

- **Two-Phase Development Model**: Start with natural language prototyping, progressively implement deterministic code
- **Advanced RAG Capabilities**: GraphRAG, HyDE, Self-RAG, CRAG, and HybridRAG implementations
- **Semantic Caching**: Achieve 31% cache hit rate for similar queries
- **Model Context Protocol (MCP)**: Standardized integration with AI assistants
- **Comprehensive Dependency System**: Manage features, services, models, and containers declaratively
- **Redis Vector Database**: Powered by Redis v8 for high-performance vector operations
- **Dual-Form Architecture**: Operates as both CLI tool and MCP server

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [File Format](#file-format)
- [Dependency System](#dependency-system)
- [Development Workflow](#development-workflow)
- [Architecture](#architecture)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.11+
- Redis v8 with vector search capabilities
- Docker (for containerized services)
- uv (for package management)

### Install EOL

```bash
# Clone the repository
git clone https://github.com/eoln/eol.git
cd eol

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

### Start Redis

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

## 🚀 Quick Start

### 1. Create Your First Feature

Create a file `hello-world.eol.md`:

```yaml
---
name: hello-world
version: 1.0.0
phase: prototyping
tags: [example, greeting]
---

# Hello World Feature

## Description
A simple greeting service that responds to users.

## Prototyping
```natural
When a user says hello:
  Greet them warmly
  Ask how you can help today
  Store the interaction in Redis
```
```

### 2. Run in Prototyping Mode

```bash
eol run hello-world.eol.md --phase prototyping
```

### 3. Generate Implementation

```bash
eol generate hello-world.eol.md --output src/
```

### 4. Switch to Implementation

```bash
eol switch hello-world.eol.md --to implementation
```

## 🎯 Core Concepts

### Two-Phase Development Model

EOL enables development through two complementary phases:

#### Phase 1: Prototyping
- Write features in natural language
- Execute via LLMs and MCP servers
- Rapid iteration and experimentation
- No code compilation required

#### Phase 2: Implementation
- Convert validated prototypes to deterministic code
- Optimize performance and reliability
- Deploy to production
- Maintain test coverage

#### Hybrid Mode
- Mix prototyping and implementation
- Gradual transition between phases
- Per-operation phase control

### File Format

EOL uses Markdown-based files with embedded code:

- **`.eol.md`** - Feature specifications
- **`.test.eol.md`** - Test specifications with Gherkin support

Example structure:
```yaml
---
name: feature-name
version: 1.0.0
phase: hybrid
dependencies:
  models:
    - name: claude-3-opus
      provider: anthropic
      purpose: reasoning
---

# Feature content...
```

## 🔗 Dependency System

EOL provides comprehensive dependency management across six types:

### 1. Feature Dependencies
```yaml
dependencies:
  features:
    - path: auth/authentication.eol.md
      version: "^2.0.0"
      inject: [authenticate_user]
```

### 2. MCP Server Dependencies
```yaml
dependencies:
  mcp_servers:
    - name: redis-mcp
      version: ">=1.0.0"
      transport: stdio
```

### 3. Service Dependencies
```yaml
dependencies:
  services:
    - name: stripe-api
      url: https://api.stripe.com
      auth:
        type: bearer
        token: ${STRIPE_API_KEY}
```

### 4. Package Dependencies
```yaml
dependencies:
  packages:
    - name: redis[vector]
      version: ">=5.0.0"
```

### 5. Container Dependencies
```yaml
dependencies:
  containers:
    - name: redis
      image: redis/redis-stack:latest
      ports: ["6379:6379"]
```

### 6. LLM Model Dependencies
```yaml
dependencies:
  models:
    - name: claude-3-opus
      provider: anthropic
      purpose: complex-reasoning
      config:
        temperature: 0.7
```

## 🔄 Development Workflow

### 1. Feature Development

```bash
# Create feature specification
eol create feature-name

# Run in prototyping mode
eol run feature-name.eol.md --phase prototyping

# Write tests
eol test feature-name.test.eol.md

# Generate implementation
eol generate feature-name.eol.md

# Switch to implementation
eol switch feature-name.eol.md --to implementation

# Deploy
eol deploy feature-name.eol.md --env production
```

### 2. Dependency Management

```bash
# Install dependencies
eol deps install

# Check dependency health
eol deps health

# Generate dependency graph
eol deps graph --output deps.svg

# Show costs (for LLM models)
eol deps cost --period month
```

### 3. Testing

```bash
# Run tests
eol test feature.test.eol.md

# Run with coverage
eol test feature.test.eol.md --coverage

# Run specific test
eol test feature.test.eol.md --pattern "payment*"
```

## 🏗 Architecture

### System Layers

```
┌─────────────────────────────────────────────────┐
│                 EOL CLI / MCP Server            │
├─────────────────────────────────────────────────┤
│                  Core Engine                     │
│  (Parser, Phase Manager, Context Manager)       │
├─────────────────────────────────────────────────┤
│                Execution Layer                   │
│  (Prototyping Engine, Implementation Engine)    │
├─────────────────────────────────────────────────┤
│                 Service Layer                    │
│  (MCP Services, Redis Vector, Code Generator)   │
├─────────────────────────────────────────────────┤
│              Infrastructure Layer                │
│  (Redis v8, Docker, External APIs)              │
└─────────────────────────────────────────────────┘
```

### Monorepo Structure

```
eol/
├── .claude/                  # AI context management
│   └── context/             # Documentation
├── packages/                # Core packages
│   ├── eol-core/           # Core engine
│   ├── eol-proto/          # Prototyping engine
│   ├── eol-impl/           # Implementation engine
│   ├── eol-redis/          # Redis integration
│   ├── eol-mcp/            # MCP services
│   └── eol-cli/            # CLI interface
├── examples/               # Example features
├── features/               # Feature specifications
└── tests/                  # Test suites
```

## 📚 Examples

### Payment Processor
Complete payment processing system with fraud detection:
```bash
cd examples
eol run payment-processor.eol.md
```

### RAG Implementation
Advanced RAG with semantic caching:
```bash
cd examples
eol run rag-system.eol.md
```

### Real-time Chat
WebSocket-based chat with Redis Pub/Sub:
```bash
cd examples
eol run chat-service.eol.md
```

## 📖 Documentation

Comprehensive documentation is available in `.claude/context/`:

- [MCP Architecture](/.claude/context/mcp-architecture.md)
- [Redis Vector Database](/.claude/context/redis-vector-db.md)
- [Two-Phase Development](/.claude/context/eol-phases.md)
- [RAG Patterns](/.claude/context/rag-patterns.md)
- [Dependency System](/.claude/context/eol-dependencies.md)
- [File Format Specification](/.claude/context/eol-file-format.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/eoln/eol.git
cd eol

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy .
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude and MCP
- [Redis](https://redis.io) for vector database capabilities
- [FastMCP](https://github.com/jlowin/fastmcp) for MCP implementation
- The open-source community for continuous inspiration

## 🔗 Links

- [Documentation](https://docs.eol.dev)
- [GitHub Repository](https://github.com/eoln/eol)
- [Issue Tracker](https://github.com/eoln/eol/issues)
- [Discord Community](https://discord.gg/eol)

---

Built with ❤️ for the AI development community