# Context Directory Rules

## Core Principles
- ALWAYS read existing files before making changes
- Never assume - always verify
- Update immediately with discoveries

## Usage Patterns
- Load domain-specific subdirectories for specialized tasks
- Check multiple contexts for complex operations
- Keep examples from actual codebase only

## Directory Structure
- **rag/**: RAG-specific patterns and metrics
- **redis/**: Redis best practices and patterns
- **python/**: Python standards and conventions
- Individual `.md` files for cross-cutting concerns

## Context Loading Strategy
1. Load general context files first
2. Load domain-specific context as needed
3. Combine multiple contexts for complex tasks
4. Prefer specific over general context

## Maintenance
- Update context based on lessons learned
- Keep examples current with codebase
- Remove outdated patterns
- Document anti-patterns discovered