# .claude Directory Rules

## Context Engineering Principles
- WRITE: Use plans/ for persistent task tracking
- SELECT: Load context/ files based on task requirements  
- COMPRESS: Keep context window <80% utilization
- ISOLATE: Respect directory boundaries

## Directory Navigation
- Always check for nested CLAUDE.md when entering subdirectories
- Subdirectory rules override parent rules
- Load relevant context before ANY operation

## Directory Structure
- **context/**: Domain-specific knowledge and patterns
- **plans/**: Task planning and tracking with state management
- **commands/**: Reusable command patterns and workflows
- **settings.local.json**: MCP server configuration

## Usage
When working in this directory:
1. Check for local CLAUDE.md files in subdirectories
2. Load appropriate context based on task type
3. Use plans/ for structured task management
4. Reference commands/ for common workflows