# Redis MCP Servers Analysis

## Overview

Redis MCP servers provide natural language interfaces for AI agents to interact with Redis databases through the Model Context Protocol (MCP).

## Official Redis MCP Server

### Repository

- **GitHub**: <https://github.com/redis/mcp-redis>
- **Maintained by**: Redis official team
- **Language**: Python

### Key Features

#### Natural Language Interface

- AI agents can query and update Redis using natural language
- Seamless integration with MCP clients
- Structured and unstructured data interaction

#### Data Type Support

- **Strings**: Set/get with expiration for configuration, sessions, caching
- **Hashes**: Field-value pairs, vector embeddings, user profiles
- **Lists**: Append/pop operations for queues, message brokers
- **Sets**: Unique value tracking (user IDs, tags)
- **Sorted Sets**: Ranked data management
- **JSON Documents**: Complex structured data
- **Streams**: Event streaming
- **Vector Embeddings**: AI/ML data storage

#### Server Management

- Health check tools
- Database status monitoring
- Performance metrics

### Installation & Configuration

#### Using uvx (Recommended)

```bash
# Basic connection
uvx --from git+https://github.com/redis/mcp-redis.git@0.2.0 redis-mcp-server \
  --host localhost \
  --port 6379 \
  --password mypassword

# Using Redis URI
uvx --from git+https://github.com/redis/mcp-redis.git redis-mcp-server \
  --url redis://user:pass@localhost:6379/0

# SSL connection
uvx --from git+https://github.com/redis/mcp-redis.git redis-mcp-server \
  --url rediss://user:pass@redis.example.com:6379/0
```

#### Configuration Methods

1. Command line arguments (highest precedence)
2. Environment variables
3. Default values

### Transport Support

- **stdio**: Local subprocess communication
- **SSE**: Network availability (Server-Sent Events)
- **streamable-http**: Stateful requests without persistent connections

### Integration Points

#### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "Redis MCP Server": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/redis/mcp-redis.git",
        "redis-mcp-server",
        "--url",
        "redis://localhost:6379/0"
      ]
    }
  }
}
```

#### Supported Platforms

- Claude Desktop
- Cursor IDE
- VS Code with GitHub Copilot
- OpenAI Agents SDK
- Custom MCP clients

## Redis Cloud MCP Server

### Repository

- **GitHub**: <https://github.com/redis/mcp-redis-cloud>
- **Purpose**: Manage Redis Cloud resources via natural language

### Features

- Direct access to Redis Cloud subscription
- Database management without extra tooling
- Cloud resource provisioning
- Scaling operations

## Alternative Implementations

### mcp-server-redis (Community)

- **GitHub**: <https://github.com/prajwalnayak7/mcp-server-redis>
- **Features**: AWS MemoryDB support, modular architecture
- **Structure**:
  - Configuration module
  - Connection management
  - Resources (status, keys)
  - Tools (operations, lists, hashes, sets, pub/sub)

### REDIS-MCP-Server (Community)

- **GitHub**: <https://github.com/GongRzhe/REDIS-MCP-Server>
- **Features**: Standardized tools for key-value operations
- **Focus**: LLM interaction with Redis stores

### redis-mcp-server (Spring Boot)

- **GitHub**: <https://github.com/yyue9527/redis-mcp-server>
- **Stack**: Spring Boot + Spring AI
- **Language**: Java

## Python Implementation Tools

### MCP Utils Package

```python
from mcp_utils.core import MCPServer
from mcp_utils.schema import GetPromptResult, Message, TextContent

# Create MCP server
mcp = MCPServer("redis-tools", "1.0")

@mcp.prompt()
def cache_lookup_prompt(key: str) -> GetPromptResult:
    return GetPromptResult(
        description="Cache lookup prompt",
        messages=[
            Message(
                role="user",
                content=TextContent(
                    text=f"Retrieve cached value for key: {key}",
                ),
            )
        ],
    )

@mcp.tool()
def get_cached_value(key: str) -> str:
    # Redis retrieval logic
    return redis_client.get(key)
```

### OpenAI Agents SDK Integration

```python
from openai_agents import Agent
from mcp_redis import RedisMCPClient

# Initialize Redis MCP client
redis_mcp = RedisMCPClient(url="redis://localhost:6379")

# Create agent with Redis tools
agent = Agent(
    tools=[redis_mcp.get_tools()],
    resources=[redis_mcp.get_resources()]
)
```

## Use Cases for EOL Integration

### Primary Use Cases

1. **Session Management**: User conversation history
2. **Context Storage**: LLM context persistence
3. **Real-time Caching**: Semantic response caching
4. **Rate Limiting**: API usage control
5. **Recommendations**: Vector similarity search
6. **RAG Implementation**: Document retrieval and indexing

### Integration Strategy for EOL

#### As Direct Dependency

```python
# pyproject.toml
[tool.poetry.dependencies]
redis-mcp = {git = "https://github.com/redis/mcp-redis.git", tag = "0.2.0"}
```

#### As Subprocess Service

```python
import subprocess
import json

class RedisMCPService:
    def __init__(self, redis_url):
        self.process = subprocess.Popen(
            ["uvx", "--from", "git+https://github.com/redis/mcp-redis.git",
             "redis-mcp-server", "--url", redis_url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def query(self, prompt):
        # Send natural language query to MCP server
        request = json.dumps({"prompt": prompt})
        self.process.stdin.write(request.encode())
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())
```

#### As Docker Service

```dockerfile
FROM python:3.11-slim
RUN pip install uvx
CMD ["uvx", "--from", "git+https://github.com/redis/mcp-redis.git", \
     "redis-mcp-server", "--url", "${REDIS_URL}"]
```

## Recommendations for EOL

### Architecture Decision

1. **Use redis-mcp as a subsystem** for natural language Redis operations
2. **Extend with custom tools** for EOL-specific operations
3. **Leverage existing tools** for standard Redis operations
4. **Build on top** rather than reimplementing

### Implementation Approach

```python
# eol/services/redis_mcp.py
from redis_mcp import RedisMCPServer
from fastmcp import FastMCP

class EOLRedisMCP(RedisMCPServer):
    """Extended Redis MCP for EOL framework"""

    def __init__(self, redis_config):
        super().__init__(redis_config)
        self.add_eol_tools()

    def add_eol_tools(self):
        @self.tool()
        async def store_context(context_id: str, content: str, embedding: list):
            """Store EOL context with embeddings"""
            # Custom implementation
            pass

        @self.tool()
        async def retrieve_relevant_context(query: str, k: int = 5):
            """Retrieve relevant context using vector search"""
            # Custom implementation
            pass
```

### Benefits of Integration

1. **Reduced Development Time**: Leverage existing Redis MCP implementation
2. **Natural Language Interface**: Built-in NL processing for Redis operations
3. **Maintained by Redis**: Official support and updates
4. **Community Ecosystem**: Access to extensions and integrations
5. **Protocol Compliance**: Full MCP specification support

## Conclusion

Integrating redis-mcp as an EOL subsystem provides a robust foundation for Redis operations while allowing custom extensions for EOL-specific requirements. This approach maximizes code reuse and ensures compatibility with the broader MCP ecosystem.
