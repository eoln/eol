# EOL Dual-Form Architecture: CLI and MCP Server

## Overview
EOL operates as both a command-line interface (CLI) tool for developers and a Model Context Protocol (MCP) server for AI assistants, providing maximum flexibility for human and AI-driven workflows.

## Architecture Design

### Core Principle
Single codebase serving two interfaces:
- **CLI Mode**: Direct developer interaction
- **MCP Mode**: AI assistant integration (Claude, Cursor, etc.)

```
┌─────────────────────────────────────────────────┐
│                EOL Framework                    │
├─────────────────┬───────────────────────────────┤
│    CLI Interface│      MCP Interface            │
│    (Typer)      │      (FastMCP)                │
├─────────────────┴───────────────────────────────┤
│              Shared Core Logic                  │
│  (Parser, Executor, Context Manager, etc.)      │
└──────────────────────────────────────────────────┘
```

## Implementation

### Entry Point Design
```python
# eol/__main__.py
import sys
import typer
from fastmcp import FastMCP
from typing import Optional
from eol.core import FeatureExecutor, TestRunner, CodeGenerator

# Initialize both interfaces
cli = typer.Typer(name="eol", help="EOL Framework CLI")
mcp = FastMCP("eol-framework", description="EOL AI Framework MCP Server")

# Shared executor instances
executor = FeatureExecutor()
test_runner = TestRunner()
generator = CodeGenerator()

# ============= CLI Commands =============

@cli.command()
def run(
    feature: str = typer.Argument(..., help="Path to .eol.md file"),
    phase: Optional[str] = typer.Option(None, "--phase", "-p"),
    watch: bool = typer.Option(False, "--watch", "-w")
):
    """Run an EOL feature file"""
    result = executor.run(feature, phase, watch)
    display_result(result)

@cli.command()
def test(
    test_file: str = typer.Argument(..., help="Path to .test.eol.md.md file"),
    coverage: bool = typer.Option(False, "--coverage", "-c")
):
    """Run EOL tests"""
    results = test_runner.run(test_file, coverage)
    display_test_results(results)

@cli.command()
def generate(
    feature: str = typer.Argument(..., help="Feature to generate"),
    output: str = typer.Option("./src", "--output", "-o")
):
    """Generate implementation from prototype"""
    code = generator.generate(feature, output)
    print(f"Generated code in {output}")

@cli.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host")
):
    """Start EOL as MCP HTTP server"""
    mcp.run(transport="sse", host=host, port=port)
    
# ============= MCP Tools =============

@mcp.tool()
async def execute_feature(
    feature_path: str,
    phase: str = "hybrid",
    context: Optional[dict] = None
) -> dict:
    """Execute an .eol feature file
    
    Args:
        feature_path: Path to the .eol.md file
        phase: Execution phase (prototyping|implementation|hybrid)
        context: Additional execution context
        
    Returns:
        Execution results including outputs and logs
    """
    result = await executor.run_async(feature_path, phase, context)
    return {
        "status": result.status,
        "outputs": result.outputs,
        "logs": result.logs,
        "metrics": result.metrics
    }

@mcp.tool()
async def run_tests(
    test_path: str,
    coverage: bool = False,
    pattern: Optional[str] = None
) -> dict:
    """Execute .test.eol.md test specifications
    
    Args:
        test_path: Path to .test.eol.md.md file
        coverage: Generate coverage report
        pattern: Test name pattern to match
        
    Returns:
        Test results with pass/fail status
    """
    results = await test_runner.run_async(test_path, coverage, pattern)
    return {
        "passed": results.passed,
        "failed": results.failed,
        "skipped": results.skipped,
        "coverage": results.coverage if coverage else None
    }

@mcp.tool()
async def generate_implementation(
    prototype_spec: str,
    language: str = "python",
    context: Optional[dict] = None
) -> str:
    """Generate implementation code from natural language specification
    
    Args:
        prototype_spec: Natural language specification
        language: Target programming language
        context: Additional context for generation
        
    Returns:
        Generated implementation code
    """
    return await generator.generate_from_spec(
        prototype_spec, 
        language, 
        context
    )

@mcp.tool()
async def create_feature(
    name: str,
    description: str,
    requirements: list[str],
    tags: Optional[list[str]] = None
) -> str:
    """Create a new .eol feature file
    
    Args:
        name: Feature name
        description: Feature description
        requirements: List of requirements
        tags: Optional tags for categorization
        
    Returns:
        Path to created feature file
    """
    template = generator.create_feature_template(
        name, description, requirements, tags
    )
    path = f"features/{name}.eol.md"
    await save_file(path, template)
    return path

@mcp.tool()
async def switch_phase(
    feature: str,
    to_phase: str,
    operations: Optional[list[str]] = None
) -> dict:
    """Switch feature execution phase
    
    Args:
        feature: Feature name or path
        to_phase: Target phase (prototyping|implementation|hybrid)
        operations: Specific operations to switch (None = all)
        
    Returns:
        Updated phase configuration
    """
    return await executor.switch_phase(feature, to_phase, operations)

@mcp.tool()
async def analyze_performance(
    feature: str,
    metrics: list[str] = ["latency", "throughput", "cache_hits"],
    time_range: Optional[str] = None
) -> dict:
    """Analyze feature performance metrics
    
    Args:
        feature: Feature name or path
        metrics: Metrics to analyze
        time_range: Time range for analysis (e.g., "1h", "24h")
        
    Returns:
        Performance analysis report
    """
    from eol.monitoring import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer()
    return await analyzer.analyze(feature, metrics, time_range)

@mcp.tool()
async def search_context(
    query: str,
    limit: int = 10,
    doc_types: Optional[list[str]] = None
) -> list[dict]:
    """Search through EOL context using semantic similarity
    
    Args:
        query: Search query
        limit: Maximum results to return
        doc_types: Filter by document types
        
    Returns:
        List of relevant context documents
    """
    from eol.context import ContextSearcher
    searcher = ContextSearcher()
    return await searcher.search(query, limit, doc_types)

# ============= MCP Resources =============

@mcp.resource("eol://features/{feature_name}")
async def get_feature_info(feature_name: str) -> dict:
    """Get detailed feature information
    
    Returns feature metadata, status, and configuration
    """
    from eol.registry import FeatureRegistry
    registry = FeatureRegistry()
    return await registry.get_feature_info(feature_name)

@mcp.resource("eol://context/{category}/{doc_name}")
async def get_context_document(category: str, doc_name: str) -> str:
    """Retrieve context documentation
    
    Categories: architecture, patterns, examples, guides
    """
    path = f".claude/context/{category}/{doc_name}.md"
    return await read_file(path)

@mcp.resource("eol://metrics/{feature_name}")
async def get_feature_metrics(feature_name: str) -> dict:
    """Get real-time feature metrics
    
    Returns performance metrics, usage stats, and health status
    """
    from eol.monitoring import MetricsCollector
    collector = MetricsCollector()
    return await collector.get_metrics(feature_name)

@mcp.resource("eol://config")
async def get_configuration() -> dict:
    """Get current EOL configuration
    
    Returns active configuration including phase settings,
    Redis connection, and enabled features
    """
    from eol.config import ConfigManager
    config = ConfigManager()
    return config.get_active_config()

# ============= MCP Prompts =============

@mcp.prompt("create_ai_feature")
async def prompt_create_ai_feature() -> str:
    """Template for creating an AI-powered feature"""
    return """Create an AI feature with the following structure:
    
    1. Feature Specification:
       - Name and description
       - Natural language prototype
       - Requirements and constraints
    
    2. Test Specifications:
       - Gherkin scenarios for validation
       - Performance requirements
       - Edge cases
    
    3. Implementation Plan:
       - Start with prototyping phase
       - Progressive implementation steps
       - Integration points
    
    4. Configuration:
       - Redis backend settings
       - Caching strategy
       - Monitoring setup
    """

@mcp.prompt("optimize_rag")
async def prompt_optimize_rag() -> str:
    """Guide for optimizing RAG implementation"""
    return """Optimize your RAG implementation:
    
    1. Analyze Current Setup:
       - Chunking strategy effectiveness
       - Retrieval accuracy metrics
       - Query latency
    
    2. Implement Improvements:
       - Semantic caching (31% hit rate target)
       - Hybrid search (vector + keyword)
       - Smart chunking by content type
    
    3. Configure Advanced Features:
       - GraphRAG for relationships
       - HyDE for query expansion
       - Self-RAG for quality control
    
    4. Monitor and Iterate:
       - Set up performance tracking
       - A/B test configurations
       - Continuous optimization
    """

@mcp.prompt("test_driven_development")
async def prompt_tdd() -> str:
    """Guide for test-driven EOL development"""
    return """Test-Driven Development with EOL:
    
    1. Write Test Specifications (.test.eol.md):
       - Gherkin scenarios
       - Expected behaviors
       - Edge cases
    
    2. Run Tests in Prototype Mode:
       - Natural language execution
       - Validate requirements
    
    3. Generate Implementation:
       - Convert passing tests to code
       - Maintain test coverage
    
    4. Refactor and Optimize:
       - Performance improvements
       - Code quality enhancements
    """

# ============= Main Entry Logic =============

def main():
    """Main entry point with mode detection"""
    
    # Check for MCP mode indicators
    if "--mcp" in sys.argv or "MCP_MODE" in os.environ:
        # Running as MCP server
        transport = os.environ.get("MCP_TRANSPORT", "stdio")
        
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "sse":
            port = int(os.environ.get("MCP_PORT", 8000))
            host = os.environ.get("MCP_HOST", "0.0.0.0")
            mcp.run(transport="sse", host=host, port=port)
    else:
        # Running as CLI
        cli()

if __name__ == "__main__":
    main()
```

## Integration Configurations

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "eol": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/eol",
        "run",
        "python",
        "-m",
        "eol",
        "--mcp"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

### Cursor/IDE Integration
```json
{
  "mcp": {
    "servers": {
      "eol": {
        "url": "http://localhost:8000/mcp",
        "transport": "sse"
      }
    }
  }
}
```

### Docker Configuration

#### Dockerfile for Dual Mode
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

# Copy application code
COPY . .

# Create volume mount points
VOLUME ["/features", "/data", "/.claude/context"]

# Expose port for HTTP mode
EXPOSE 8000

# Default to MCP stdio mode
ENV MCP_MODE=true
ENV MCP_TRANSPORT=stdio

# Flexible entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "eol"]

# Default command (can be overridden)
CMD ["--mcp"]
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  # EOL as CLI tool
  eol-cli:
    build: .
    environment:
      MCP_MODE: "false"
    volumes:
      - ./features:/features
      - ./data:/data
      - ./.claude/context:/.claude/context
    command: ["run", "features/example.eol.md", "--watch"]
  
  # EOL as MCP HTTP server
  eol-mcp-http:
    build: .
    environment:
      MCP_MODE: "true"
      MCP_TRANSPORT: "sse"
      MCP_PORT: "8000"
      MCP_HOST: "0.0.0.0"
    ports:
      - "8000:8000"
    volumes:
      - ./features:/features
      - ./data:/data
      - ./.claude/context:/.claude/context
  
  # EOL as MCP stdio server (for testing)
  eol-mcp-stdio:
    build: .
    environment:
      MCP_MODE: "true"
      MCP_TRANSPORT: "stdio"
    volumes:
      - ./features:/features
      - ./data:/data
    stdin_open: true
    tty: true
```

## Usage Examples

### CLI Mode Examples
```bash
# Run a feature
eol run user-auth.eol.md --phase prototyping

# Run tests
eol test user-auth.test.eol.md --coverage

# Generate implementation
eol generate user-auth.eol.md --output src/

# Start MCP HTTP server
eol serve --port 8000

# Watch mode for development
eol run user-auth.eol.md --watch
```

### MCP Mode Examples

#### From Claude Desktop
```
Human: Can you help me create a new authentication feature?