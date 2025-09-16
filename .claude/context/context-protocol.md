# Context Protocol Methodology

## Overview

Context Protocol encompasses methodologies for managing LLM context effectively, including Model Context Protocol (MCP), context engineering patterns, and documentation strategies like CLAUDE.md.

## Model Context Protocol (MCP)

### Core Concept

MCP is an open standard for connecting AI assistants to data sources, eliminating custom integrations between LLMs and applications.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MCP Host  │────▶│  MCP Client │────▶│  MCP Server │
│(Claude/IDE) │     │  (Protocol) │     │   (Data)    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Implementation Pattern

```python
# MCP Server Example
from mcp import Server, Resource, Tool

class ContextServer(Server):
    """MCP server for context management"""

    @Resource("project_context")
    async def get_context(self):
        """Expose project context as resource"""
        return {
            "files": self.scan_project_files(),
            "documentation": self.load_documentation(),
            "configuration": self.load_config()
        }

    @Tool("search_context")
    async def search(self, query: str):
        """Search through project context"""
        results = await self.vector_search(query)
        return self.format_results(results)
```

## Context Engineering Methodology

### Definition

Context engineering is the "delicate art and science of filling the context window with just the right information for the next step."

### Core Strategies

#### 1. Write - Creating Context

```python
class ContextWriter:
    def create_context_file(self, project_path):
        """Generate context documentation"""

        # CLAUDE.md - Project-wide rules
        claude_md = self.generate_claude_md(project_path)

        # .context directory structure
        context_structure = {
            ".context/": {
                "overview.md": self.project_overview(),
                "architecture.md": self.architecture_docs(),
                "patterns.md": self.code_patterns(),
                "examples/": self.gather_examples()
            }
        }

        return context_structure
```

#### 2. Select - Choosing Relevant Context

```python
class ContextSelector:
    def select_for_task(self, task, available_context):
        """Select relevant context for specific task"""

        selected = {
            "episodic": [],  # Examples of desired behavior
            "procedural": [], # Instructions to steer behavior
            "semantic": []    # Task-relevant facts
        }

        # Episodic memories - few-shot examples
        if task.needs_examples:
            selected["episodic"] = self.find_similar_examples(
                task,
                available_context["examples"]
            )

        # Procedural memories - instructions
        selected["procedural"] = self.get_relevant_instructions(
            task.type,
            available_context["instructions"]
        )

        # Semantic memories - facts
        selected["semantic"] = self.get_relevant_facts(
            task.domain,
            available_context["knowledge"]
        )

        return selected
```

#### 3. Compress - Optimizing Context

```python
class ContextCompressor:
    def auto_compact(self, context, window_size):
        """Compress context when approaching limits"""

        if self.get_usage(context) > window_size * 0.95:
            # Recursive summarization
            compressed = self.recursive_summarize(context)

            # Hierarchical summarization
            hierarchical = self.hierarchical_summarize(context)

            # Choose best compression
            return self.select_best_compression(
                compressed,
                hierarchical,
                window_size
            )

        return context
```

#### 4. Isolate - Managing Boundaries

```python
class ContextIsolator:
    def isolate_contexts(self, tasks):
        """Manage context boundaries between tasks"""

        isolated_contexts = {}

        for task in tasks:
            # Create isolated context
            isolated_contexts[task.id] = {
                "scope": task.scope,
                "dependencies": task.dependencies,
                "isolation_level": task.isolation_level
            }

            # Prevent context bleeding
            if task.isolation_level == "strict":
                isolated_contexts[task.id]["boundary"] = "hard"
            else:
                isolated_contexts[task.id]["boundary"] = "soft"

        return isolated_contexts
```

## CLAUDE.md Methodology

### Purpose

CLAUDE.md serves as project-wide rules that AI assistants follow in every conversation.

### Structure

```markdown
# Project Name

## Overview
Brief description of the project and its goals.

## Coding Standards
- Language-specific conventions
- Naming conventions
- File organization

## Architecture Decisions
- Key architectural patterns
- Technology choices
- Design principles

## AI Assistant Guidelines
- How to approach tasks
- What to prioritize
- Common patterns to follow

## Examples
Links to example implementations in `.context/examples/`

## Do's and Don'ts
- DO: Follow existing patterns
- DON'T: Introduce new dependencies without discussion
```

### Best Practices

```python
class CLAUDEmdGenerator:
    def generate(self, project):
        """Generate CLAUDE.md from project analysis"""

        sections = {
            "overview": self.analyze_project_structure(project),
            "standards": self.extract_coding_standards(project),
            "patterns": self.identify_patterns(project),
            "examples": self.collect_examples(project),
            "guidelines": self.create_guidelines(project)
        }

        return self.format_markdown(sections)

    def update_incrementally(self, existing_claude_md, changes):
        """Update CLAUDE.md based on project evolution"""

        # Parse existing
        current = self.parse_markdown(existing_claude_md)

        # Apply changes
        for change in changes:
            if change.type == "new_pattern":
                current["patterns"].append(change.content)
            elif change.type == "updated_standard":
                current["standards"][change.key] = change.value

        return self.format_markdown(current)
```

## .context Directory Structure

### Standard Layout

```
.context/
├── overview.md           # Project overview
├── architecture.md       # Architecture documentation
├── patterns.md          # Code patterns and conventions
├── examples/            # Example implementations
│   ├── feature_a.md
│   └── feature_b.md
├── knowledge/           # Domain knowledge
│   ├── business_rules.md
│   └── technical_specs.md
└── memory/              # Persistent memories
    ├── decisions.md     # Architecture decisions
    └── learnings.md     # Lessons learned
```

### Implementation

```python
class ContextDirectory:
    def initialize(self, project_path):
        """Initialize .context directory structure"""

        context_path = Path(project_path) / ".context"

        # Create directory structure
        directories = [
            "examples",
            "knowledge",
            "memory"
        ]

        for dir in directories:
            (context_path / dir).mkdir(parents=True, exist_ok=True)

        # Generate initial files
        self.create_overview(context_path)
        self.create_architecture(context_path)
        self.create_patterns(context_path)

        return context_path

    def update_context(self, event):
        """Update context based on project events"""

        if event.type == "code_change":
            self.update_patterns(event.changes)
        elif event.type == "decision":
            self.record_decision(event.decision)
        elif event.type == "learning":
            self.record_learning(event.learning)
```

## Context Management Tools

### LLM Context.py Pattern

```python
class LLMContext:
    """Tool for injecting relevant content into LLM context"""

    def __init__(self, project_path):
        self.project = project_path
        self.gitignore = self.load_gitignore()

    def collect_context(self, task):
        """Collect relevant files for task"""

        # Smart file selection using .gitignore
        files = self.select_files(
            self.project,
            exclude_patterns=self.gitignore
        )

        # Filter by relevance
        relevant = self.filter_by_relevance(files, task)

        # Format for LLM
        return self.format_for_llm(relevant)

    def to_clipboard(self, context):
        """Copy context to clipboard for manual paste"""
        formatted = self.format_context(context)
        pyperclip.copy(formatted)

    def to_mcp(self, context):
        """Send context via MCP"""
        return self.mcp_client.send_context(context)
```

### Vector-Based Context Search

```python
class VectorContextSearch:
    """Search through codebase context using embeddings"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.index_name = "context_search"

    async def index_codebase(self, path):
        """Index entire codebase for context search"""

        for file_path in self.walk_files(path):
            # Parse file
            content = self.parse_file(file_path)

            # Chunk based on file type
            chunks = self.chunk_by_type(content, file_path)

            # Store in vector DB
            for chunk in chunks:
                embedding = await self.embed(chunk.content)

                await self.redis.hset(f"context:{chunk.id}", {
                    "file": str(file_path),
                    "content": chunk.content,
                    "embedding": embedding.tobytes(),
                    "type": chunk.type,
                    "metadata": json.dumps(chunk.metadata)
                })

    async def search(self, query, limit=10):
        """Search for relevant context"""

        query_embedding = await self.embed(query)

        results = await self.redis.ft(self.index_name).search(
            Query("*=>[KNN {} @embedding $vec]".format(limit))
            .dialect(2),
            query_params={"vec": query_embedding.tobytes()}
        )

        return self.format_results(results)
```

## Context Window Management

### Strategies

```python
class ContextWindowManager:
    def __init__(self, max_tokens=100000):
        self.max_tokens = max_tokens
        self.usage = 0

    def manage_window(self, new_context):
        """Manage context window efficiently"""

        strategies = {
            "below_50": self.add_liberally,
            "50_to_75": self.add_selectively,
            "75_to_90": self.compress_and_add,
            "above_90": self.replace_least_relevant
        }

        usage_percent = (self.usage / self.max_tokens) * 100

        if usage_percent < 50:
            return strategies["below_50"](new_context)
        elif usage_percent < 75:
            return strategies["50_to_75"](new_context)
        elif usage_percent < 90:
            return strategies["75_to_90"](new_context)
        else:
            return strategies["above_90"](new_context)
```

## EOL Framework Integration

### Context Protocol in EOL

```yaml
# context-config.eol
name: context-management
phase: implementation

context:
  protocol: mcp
  storage: redis

  structure:
    root: .context
    claude_md: true
    examples: true

  strategies:
    - vector_search
    - smart_selection
    - auto_compression

  mcp_servers:
    - name: eol-context
      type: context_provider
      resources:
        - project_files
        - documentation
        - examples
      tools:
        - search_context
        - update_context
```

### Implementation

```python
class EOLContextManager:
    """EOL-specific context management"""

    def __init__(self, project_path):
        self.project = project_path
        self.context_dir = Path(project_path) / ".context"
        self.claude_md = Path(project_path) / "CLAUDE.md"

    async def initialize(self):
        """Initialize EOL context structure"""

        # Create .context directory
        self.context_dir.mkdir(exist_ok=True)

        # Generate CLAUDE.md
        await self.generate_claude_md()

        # Set up MCP server
        await self.setup_mcp_server()

        # Index existing code
        await self.index_codebase()

    async def maintain_context(self):
        """Maintain context as project evolves"""

        # Watch for changes
        async for change in self.watch_changes():
            if change.affects_context():
                await self.update_context(change)

            # Auto-compact if needed
            if await self.should_compact():
                await self.compact_context()
```

## Best Practices

1. **Keep CLAUDE.md Updated**: Regularly update as project evolves
2. **Use Semantic Organization**: Group related context together
3. **Implement Auto-Compression**: Prevent context overflow
4. **Leverage MCP**: Use standardized protocol for integrations
5. **Version Context**: Track context evolution with project
6. **Test Context Quality**: Validate context relevance regularly
7. **Isolate Sensitive Data**: Keep secrets out of context files
