# MCP (Model Context Protocol) Architecture

## Overview

MCP is a standardized protocol that allows applications to provide context for LLMs, separating the concerns of providing context from the actual LLM interaction.

## Architecture Components

### 1. Core Architecture

- **MCP Server**: Bridge/API between MCP world and external system functionality
- **MCP Client**: Consumes server capabilities and interfaces with LLMs
- **MCP Host**: Runtime environment managing client-server communication (e.g., Claude Desktop, Cursor, custom agents)

### 2. Server Capabilities

#### Resources (GET-like endpoints)

- Load information into LLM's context
- Expose data from external systems
- Read-only operations

#### Tools (POST-like endpoints)

- Execute code or produce side effects
- Provide functionality to LLMs
- Action-oriented operations

#### Prompts

- Reusable templates for LLM interactions
- Standardized interaction patterns
- Context-aware prompt management

### 3. Communication Methods

#### stdio (Standard Input/Output)

- Used for local client-server communication
- Simple and effective for local integrations
- Accessing local files or running local scripts

#### HTTP via SSE (Server-Sent Events)

- Remote client-server communication
- Scalable for distributed systems
- Real-time event streaming

## Python Implementation

### SDK Setup

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
```

### Dependencies

- `mcp[cli]>=1.4.0` - Core MCP server functionality
- `requests>=2.31.0` - API calls
- `python-dotenv>=1.0.0` - Environment variables
- Additional integration libraries as needed

### Server Implementation Pattern

1. Initialize MCP server instance
2. Define resources for data exposure
3. Implement tools for actions
4. Configure prompts for interactions
5. Set up communication transport (stdio/HTTP)

## Python Package Recommendations for EOL

### **Recommended: FastMCP**

FastMCP is the recommended choice for the EOL framework implementation for the following reasons:

#### Advantages

- **Comprehensive MCP Support**: Covers all aspects of the MCP specification
- **Developer-Friendly**: Pythonic interface with simple decorators (like FastAPI for MCP)
- **Rich Features**:
  - Authentication hooks
  - User session management
  - Image content handling
  - Structured logging
  - Error handling
  - SSE streaming
  - Progress notifications
  - Sampling support
- **Official Integration**: FastMCP 1.0 incorporated into official MCP Python SDK
- **Active Development**: FastMCP 2.0 actively maintained with complete MCP ecosystem toolkit
- **CLI Tools**: Built-in development tools for debugging and inspection

#### Implementation Example

```python
from fastmcp import FastMCP

app = FastMCP("eol-context-server")

@app.tool()
async def manage_context(query: str) -> str:
    """Manage LLM context with Redis backend"""
    # Implementation here
    pass
```

### Alternative: Pydantic-AI-MCP

Consider Pydantic-AI-MCP if:

- Building comprehensive AI agents beyond just MCP servers
- Need integrated monitoring via Pydantic Logfire
- Working with multiple LLM providers simultaneously
- Want seamless agent-MCP integration

### Why FastMCP for EOL

1. **Protocol Focus**: EOL needs robust MCP server implementation as its core
2. **Minimal Overhead**: Direct MCP implementation without agent framework complexity
3. **Flexibility**: Easy to extend with custom Redis integrations
4. **Performance**: Lightweight and optimized for MCP protocol handling
5. **Development Speed**: Rapid prototyping with decorator-based approach

## Benefits

### Development Efficiency

- **M+N Problem Solution**: Tool creators build N servers, application developers build M clients
- **Reduced Duplication**: Single server implementation works with multiple clients
- **Vendor Flexibility**: Avoid lock-in, switch between LLM providers easily

### Composability

- Applications can be both MCP clients and servers simultaneously
- Enables layered and chained systems
- Logical distinction between client/server roles

### Language Agnostic

- Servers can be built in Python, TypeScript, Java, Rust, etc.
- Communication over standard transports
- Cross-language interoperability

## Integration Points

### Supported Platforms

- Claude Desktop
- Cursor IDE
- VS Code with GitHub Copilot
- OpenAI Agents SDK
- Custom agent implementations

### Full Implementation Languages

- Python SDK (complete)
- TypeScript SDK (complete)
- Other languages via protocol implementation

## Best Practices

1. **Server Design**
   - Keep servers focused on specific domains
   - Implement proper error handling
   - Use appropriate transport for use case

2. **Resource Management**
   - Cache frequently accessed data
   - Implement pagination for large datasets
   - Version resources appropriately

3. **Tool Implementation**
   - Validate inputs thoroughly with Pydantic models
   - Provide clear tool descriptions
   - Handle side effects responsibly
   - Use process_tool_call for metadata injection

4. **Security Considerations**
   - Authenticate client connections
   - Validate and sanitize inputs
   - Implement rate limiting
   - Audit tool usage
   - Security as paramount concern for AI-external system integration

## EOL Framework Integration

For the EOL framework, MCP (via FastMCP) will serve as:

- Primary protocol for context-oriented services
- Bridge between LLMs and Redis context storage
- Interface for executing .eol file scripts
- Communication layer for distributed RAG components
- Foundation for real-time context management system
