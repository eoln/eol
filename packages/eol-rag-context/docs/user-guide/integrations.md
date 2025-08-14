# MCP Integration

Connect EOL RAG Context with Claude Desktop and other Model Context Protocol (MCP) clients for seamless intelligent context retrieval in your conversations and workflows.

## Overview

MCP (Model Context Protocol) enables standardized communication between AI assistants and context providers. EOL RAG Context implements MCP as a server, allowing Claude Desktop and other MCP clients to access your indexed knowledge base naturally through conversation.

### Key Benefits

- **Seamless Integration**: No API calls or manual searches - just ask Claude
- **Contextual Awareness**: Claude automatically finds relevant information
- **Real-time Access**: Live connection to your indexed knowledge
- **Natural Interaction**: Conversational interface to your data
- **Extensible**: Works with any MCP-compatible client

## Claude Desktop Integration

### Installation and Setup

Complete setup for Claude Desktop integration:

**1. Install EOL RAG Context:**

```bash
pip install eol-rag-context
```

**2. Create Configuration File:**

```yaml
# eol-rag-config.yaml
redis:
  url: "redis://localhost:6379"
  db: 0

embedding:
  provider: "sentence_transformers"
  model: "all-MiniLM-L6-v2"

indexing:
  chunk_size: 1000
  chunk_overlap: 200

caching:
  enabled: true
  ttl_seconds: 3600
  target_hit_rate: 0.31

context:
  max_context_size: 8000
```

**3. Configure Claude Desktop:**

Find your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

Add EOL RAG Context as an MCP server:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "eol-rag-context",
      "args": ["serve", "--config", "/path/to/eol-rag-config.yaml"],
      "env": {
        "EOL_REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

**4. Start Redis Stack:**

```bash
# Using Docker
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Or using Homebrew (macOS)
brew services start redis-stack
```

**5. Restart Claude Desktop:**
Close and reopen Claude Desktop to load the MCP server.

### Verification

Test the integration:

**1. Check MCP Server Status:**
In Claude Desktop, start a new conversation and ask:

```
Can you check what MCP tools are available?
```

Claude should show EOL RAG Context tools like:

- `index_directory` - Index documents
- `search_context` - Search knowledge base
- `query_knowledge_graph` - Explore relationships

**2. Index Your First Documents:**

```
Can you index the documents in /path/to/my/project?
```

**3. Search Your Knowledge:**

```
What does the UserService class do in my codebase?
```

## MCP Tools Reference

### Available Tools

EOL RAG Context provides these MCP tools:

#### `index_directory`

Index documents in a directory:

```json
{
  "directory_path": "/path/to/documents",
  "recursive": true,
  "file_patterns": ["*.py", "*.md", "*.json"],
  "exclude_patterns": ["*.pyc", "__pycache__/*"],
  "force_reindex": false
}
```

#### `search_context`

Search indexed content:

```json
{
  "query": "how to implement authentication",
  "max_results": 5,
  "similarity_threshold": 0.7,
  "search_level": "auto"
}
```

#### `query_knowledge_graph`

Explore knowledge relationships:

```json
{
  "query": "authentication dependencies",
  "max_depth": 2,
  "include_relationships": true,
  "entity_types": ["class", "function"]
}
```

#### `get_indexing_stats`

View indexing statistics:

```json
{
  "include_performance": true,
  "include_file_breakdown": true
}
```

### Natural Language Usage

Instead of using tools directly, interact naturally:

**Indexing Commands:**

- "Index the files in my src directory"
- "Can you index all Python and Markdown files in /project?"
- "Index the documentation folder, excluding test files"

**Search Queries:**

- "How do I configure the database connection?"
- "Show me examples of error handling in the codebase"
- "What are the main classes in the authentication module?"

**Knowledge Graph Exploration:**

- "What components depend on the UserService?"
- "Show me the relationships between auth-related classes"
- "How are the database models connected?"

## Custom MCP Client

### Building Your Own Client

Create a custom MCP client for specialized workflows:

```python
import asyncio
import json
from typing import Any, Dict
import websockets

class EOLRAGMCPClient:
    """Custom MCP client for EOL RAG Context."""

    def __init__(self, server_command: str, config_path: str):
        self.server_command = server_command
        self.config_path = config_path
        self.process = None
        self.reader = None
        self.writer = None

    async def connect(self):
        """Connect to EOL RAG MCP server."""
        import subprocess

        # Start MCP server process
        self.process = subprocess.Popen(
            [self.server_command, "serve", "--config", self.config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        self.reader = self.process.stdout
        self.writer = self.process.stdin

        # Initialize MCP connection
        await self.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "custom-eol-client",
                    "version": "1.0.0"
                }
            }
        })

        response = await self.read_response()
        print(f"MCP Server initialized: {response}")

    async def send_request(self, request: Dict[str, Any]) -> None:
        """Send request to MCP server."""
        message = json.dumps(request) + '\n'
        self.writer.write(message)
        await self.writer.drain()

    async def read_response(self) -> Dict[str, Any]:
        """Read response from MCP server."""
        line = await self.reader.readline()
        return json.loads(line.strip())

    async def index_directory(self, path: str, **kwargs) -> Dict[str, Any]:
        """Index a directory using MCP."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "index_directory",
                "arguments": {
                    "directory_path": path,
                    **kwargs
                }
            }
        }

        await self.send_request(request)
        return await self.read_response()

    async def search_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search indexed content using MCP."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_context",
                "arguments": {
                    "query": query,
                    **kwargs
                }
            }
        }

        await self.send_request(request)
        return await self.read_response()

    async def close(self):
        """Close MCP connection."""
        if self.process:
            self.process.terminate()
            await self.process.wait()

# Usage example
async def custom_client_example():
    client = EOLRAGMCPClient("eol-rag-context", "config.yaml")

    try:
        await client.connect()

        # Index documents
        print("Indexing documents...")
        index_result = await client.index_directory(
            "/path/to/project",
            recursive=True,
            file_patterns=["*.py", "*.md"]
        )
        print(f"Indexed {index_result['result']['indexed_files']} files")

        # Search for information
        print("Searching for authentication info...")
        search_result = await client.search_context(
            "user authentication implementation",
            max_results=3
        )

        for result in search_result['result']['results']:
            print(f"📄 {result['source_path']}")
            print(f"   Score: {result['similarity']:.3f}")
            print(f"   Content: {result['content'][:100]}...")

    finally:
        await client.close()

asyncio.run(custom_client_example())
```

### Workflow Integration

Integrate with development workflows:

```python
class DevelopmentWorkflow:
    """Development workflow integration with EOL RAG Context."""

    def __init__(self, mcp_client):
        self.client = mcp_client

    async def code_review_assistant(self, file_path: str):
        """Assist with code reviews using context."""
        # Search for related code patterns
        patterns_query = f"similar implementations to {file_path}"
        patterns = await self.client.search_context(patterns_query)

        # Search for relevant tests
        tests_query = f"tests for {file_path}"
        tests = await self.client.search_context(tests_query)

        # Search for documentation
        docs_query = f"documentation for {file_path}"
        docs = await self.client.search_context(docs_query)

        return {
            'similar_patterns': patterns['result']['results'],
            'related_tests': tests['result']['results'],
            'documentation': docs['result']['results']
        }

    async def feature_planning_assistant(self, feature_description: str):
        """Assist with feature planning using existing codebase."""
        # Find similar features
        similar_query = f"similar features to {feature_description}"
        similar_features = await self.client.search_context(similar_query)

        # Find relevant architecture patterns
        arch_query = f"architecture patterns for {feature_description}"
        architecture = await self.client.search_context(arch_query)

        # Find potential dependencies
        deps_query = f"dependencies needed for {feature_description}"
        dependencies = await self.client.search_context(deps_query)

        return {
            'similar_features': similar_features['result']['results'],
            'architecture_guidance': architecture['result']['results'],
            'potential_dependencies': dependencies['result']['results']
        }

    async def bug_investigation_assistant(self, error_message: str):
        """Assist with bug investigation using context."""
        # Search for similar errors
        error_query = f"similar errors to {error_message}"
        similar_errors = await self.client.search_context(error_query)

        # Search for related error handling
        handling_query = f"error handling for {error_message}"
        error_handling = await self.client.search_context(handling_query)

        # Search for debugging information
        debug_query = f"debugging {error_message}"
        debug_info = await self.client.search_context(debug_query)

        return {
            'similar_errors': similar_errors['result']['results'],
            'error_handling_patterns': error_handling['result']['results'],
            'debugging_guidance': debug_info['result']['results']
        }

# Usage in development workflow
async def workflow_example():
    client = EOLRAGMCPClient("eol-rag-context", "config.yaml")
    workflow = DevelopmentWorkflow(client)

    await client.connect()

    try:
        # Code review assistance
        review_help = await workflow.code_review_assistant("src/auth/service.py")
        print("📋 Code Review Assistance:")
        for pattern in review_help['similar_patterns'][:2]:
            print(f"   Similar: {pattern['source_path']}")

        # Feature planning assistance
        feature_help = await workflow.feature_planning_assistant("user notification system")
        print("\n🚀 Feature Planning Assistance:")
        for feature in feature_help['similar_features'][:2]:
            print(f"   Reference: {feature['source_path']}")

        # Bug investigation assistance
        bug_help = await workflow.bug_investigation_assistant("database connection timeout")
        print("\n🐛 Bug Investigation Assistance:")
        for error in bug_help['similar_errors'][:2]:
            print(f"   Similar issue: {error['source_path']}")

    finally:
        await client.close()
```

## API Integration Patterns

### Direct Python API

Use the Python API directly without MCP:

```python
import asyncio
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.server import SearchContextRequest

class DirectAPIIntegration:
    """Direct API integration patterns."""

    def __init__(self):
        self.server = None

    async def initialize(self, config_path: str = None):
        """Initialize the RAG server."""
        self.server = EOLRAGContextServer(config_path=config_path)
        await self.server.initialize()

    async def batch_query_processor(self, queries: list[str]) -> dict:
        """Process multiple queries efficiently."""
        results = {}

        # Process queries in parallel for better performance
        tasks = []
        for query in queries:
            request = SearchContextRequest(
                query=query,
                max_results=3,
                similarity_threshold=0.7
            )
            task = self.server.search_context(request, None)
            tasks.append((query, task))

        # Wait for all queries to complete
        for query, task in tasks:
            try:
                result = await task
                results[query] = result['results']
            except Exception as e:
                results[query] = {'error': str(e)}

        return results

    async def context_aware_chat(self, message: str, conversation_history: list = None) -> dict:
        """Provide context-aware chat responses."""
        # Search for relevant context based on the message
        request = SearchContextRequest(
            query=message,
            max_results=5,
            include_metadata=True
        )

        context_results = await self.server.search_context(request, None)

        # Build context for response
        relevant_context = []
        for result in context_results['results']:
            if result['similarity'] > 0.7:
                relevant_context.append({
                    'content': result['content'],
                    'source': result['source_path'],
                    'type': result['chunk_type']
                })

        return {
            'query': message,
            'relevant_context': relevant_context,
            'context_summary': self._summarize_context(relevant_context)
        }

    def _summarize_context(self, context: list) -> str:
        """Summarize context for response generation."""
        if not context:
            return "No relevant context found."

        sources = set(item['source'] for item in context)
        types = set(item['type'] for item in context)

        summary = f"Found {len(context)} relevant items from {len(sources)} sources, "
        summary += f"including {', '.join(types)}."

        return summary

    async def smart_documentation_assistant(self, topic: str) -> dict:
        """Assist with documentation using multiple search strategies."""
        # Search at different levels for comprehensive coverage
        concept_search = SearchContextRequest(
            query=topic,
            search_level='concept',
            max_results=3
        )

        detail_search = SearchContextRequest(
            query=f"{topic} implementation details",
            search_level='chunk',
            max_results=5
        )

        example_search = SearchContextRequest(
            query=f"{topic} examples usage",
            max_results=3
        )

        # Execute searches in parallel
        concept_results = await self.server.search_context(concept_search, None)
        detail_results = await self.server.search_context(detail_search, None)
        example_results = await self.server.search_context(example_search, None)

        return {
            'topic': topic,
            'conceptual_overview': concept_results['results'],
            'implementation_details': detail_results['results'],
            'usage_examples': example_results['results']
        }

# Usage examples
async def api_integration_examples():
    api = DirectAPIIntegration()
    await api.initialize("config.yaml")

    # Batch processing example
    queries = [
        "how to setup authentication",
        "database migration process",
        "error handling best practices"
    ]

    batch_results = await api.batch_query_processor(queries)
    print("📦 Batch Query Results:")
    for query, results in batch_results.items():
        if isinstance(results, list):
            print(f"   '{query}': {len(results)} results")
        else:
            print(f"   '{query}': ERROR - {results.get('error', 'unknown')}")

    # Context-aware chat example
    chat_response = await api.context_aware_chat(
        "How do I handle user authentication errors?"
    )
    print(f"\n💬 Context-Aware Response:")
    print(f"   Context items: {len(chat_response['relevant_context'])}")
    print(f"   Summary: {chat_response['context_summary']}")

    # Documentation assistant example
    doc_help = await api.smart_documentation_assistant("user management")
    print(f"\n📚 Documentation Assistant:")
    print(f"   Conceptual: {len(doc_help['conceptual_overview'])} items")
    print(f"   Details: {len(doc_help['implementation_details'])} items")
    print(f"   Examples: {len(doc_help['usage_examples'])} items")

asyncio.run(api_integration_examples())
```

## Team and Multi-User Setups

### Shared Knowledge Base

Configure for team collaboration:

```python
async def team_setup():
    """Configure EOL RAG Context for team use."""

    # Team configuration
    team_config = {
        'redis': {
            'url': 'redis://shared-redis.company.com:6379',
            'db': 0,  # Shared database
            'password': 'team-password',
        },
        'indexing': {
            # Index common repositories
            'shared_repositories': [
                '/shared/company-docs',
                '/shared/api-documentation',
                '/shared/best-practices'
            ],
            'auto_update_interval': 3600,  # Update every hour
        },
        'caching': {
            'enabled': True,
            'distributed': True,  # Share cache across team
            'max_cache_size': 5000,  # Larger cache for team
        },
        'access_control': {
            'enabled': True,
            'user_permissions': {
                'developers': ['read', 'index_personal'],
                'leads': ['read', 'write', 'index_shared'],
                'admins': ['read', 'write', 'admin', 'configure']
            }
        }
    }

    # Initialize team server
    server = EOLRAGContextServer(config=team_config)
    await server.initialize()

    print("👥 Team setup configured")

    # Set up shared indexing
    shared_paths = team_config['indexing']['shared_repositories']
    for path in shared_paths:
        result = await server.index_directory(
            path,
            recursive=True,
            shared=True,  # Mark as shared content
            auto_update=True
        )
        print(f"   Indexed shared: {path} ({result['indexed_files']} files)")

    return server

# Individual team member configuration
async def team_member_setup(user_role: str, personal_projects: list):
    """Setup for individual team member."""

    member_config = {
        'user': {
            'role': user_role,
            'personal_namespace': f"user_{user_role}",
        },
        'redis': {
            'url': 'redis://shared-redis.company.com:6379',
            'db': 0,
            'namespace': f"user_{user_role}",  # Personal namespace
        },
        'indexing': {
            'personal_projects': personal_projects,
            'auto_sync_shared': True,  # Sync shared updates
        }
    }

    server = EOLRAGContextServer(config=member_config)
    await server.initialize()

    # Index personal projects
    for project in personal_projects:
        await server.index_directory(
            project,
            personal=True,  # Mark as personal content
            recursive=True
        )

    print(f"👤 Team member setup complete for {user_role}")
    return server
```

### Access Control and Permissions

Implement access control for sensitive content:

```python
class AccessControlledRAGServer:
    """RAG server with access control."""

    def __init__(self, base_server, user_permissions):
        self.server = base_server
        self.permissions = user_permissions

    async def search_with_access_control(self, query: str, user_id: str, **kwargs):
        """Search with access control filtering."""

        # Check user permissions
        user_perms = self.permissions.get(user_id, {})

        # Add access filters to search
        access_filters = {
            'accessible_to': user_id,
            'permission_level': user_perms.get('level', 'read'),
            'exclude_sensitive': not user_perms.get('sensitive_access', False)
        }

        # Merge with existing filters
        filters = kwargs.get('filters', {})
        filters.update(access_filters)
        kwargs['filters'] = filters

        # Perform search
        request = SearchContextRequest(query=query, **kwargs)
        results = await self.server.search_context(request, None)

        # Additional filtering based on permissions
        filtered_results = []
        for result in results['results']:
            if self._can_access_result(result, user_perms):
                filtered_results.append(result)

        results['results'] = filtered_results
        return results

    def _can_access_result(self, result: dict, user_permissions: dict) -> bool:
        """Check if user can access specific result."""
        result_sensitivity = result['metadata'].get('sensitivity_level', 'public')
        user_clearance = user_permissions.get('clearance_level', 'public')

        # Define access hierarchy
        hierarchy = {'public': 0, 'internal': 1, 'confidential': 2, 'secret': 3}

        return hierarchy.get(user_clearance, 0) >= hierarchy.get(result_sensitivity, 0)
```

## Troubleshooting Integration

### Common Issues

**MCP Server Not Starting:**

```bash
# Check configuration
eol-rag-context serve --config config.yaml --validate-only

# Test MCP protocol manually
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | eol-rag-context serve --config config.yaml
```

**Claude Desktop Not Recognizing Server:**

1. Check Claude Desktop logs (usually in app data directory)
2. Verify JSON configuration syntax
3. Ensure absolute paths in configuration
4. Test server independently

**Performance Issues:**

```python
# Monitor MCP performance
async def monitor_mcp_performance():
    server = EOLRAGContextServer()
    await server.initialize()

    # Enable MCP performance monitoring
    await server.enable_mcp_monitoring(
        track_request_times=True,
        track_tool_usage=True,
        log_slow_requests=True,
        slow_threshold_ms=1000
    )

    # Get performance metrics
    metrics = await server.get_mcp_metrics()

    print("⚡ MCP Performance Metrics:")
    print(f"   Average request time: {metrics['avg_request_time_ms']:.1f}ms")
    print(f"   Slow requests: {metrics['slow_requests_count']}")
    print(f"   Most used tool: {metrics['most_used_tool']}")
    print(f"   Error rate: {metrics['error_rate']:.1%}")
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
debug_config = {
    'logging': {
        'level': 'DEBUG',
        'format': 'json',
        'include_mcp_traces': True,
        'log_file': 'eol-rag-debug.log'
    }
}
```

## Best Practices

### Integration Guidelines

**Configuration Management:**

- Use environment variables for sensitive data
- Version control configuration templates
- Document configuration options for team
- Test configurations before deployment

**Performance Optimization:**

- Enable semantic caching for better response times
- Use appropriate chunk sizes for your content
- Monitor and tune similarity thresholds
- Implement proper error handling

**Security Considerations:**

- Secure Redis instance (password, TLS)
- Implement access controls for sensitive content
- Audit search queries and access patterns
- Keep embeddings and indices secure

### Production Deployment

**For Claude Desktop Teams:**

1. Set up shared Redis instance
2. Configure team access controls
3. Establish shared indexing policies
4. Monitor usage and performance
5. Regular index maintenance and updates

**For Custom Applications:**

1. Implement proper error handling
2. Add monitoring and alerting
3. Use connection pooling for performance
4. Implement graceful degradation
5. Plan for scaling and load balancing

## Next Steps

Now that you understand integration patterns:

1. **[Examples](../examples/)** - See complete integration examples and real-world use cases
2. **[API Reference](../api-reference/)** - Detailed API documentation for all integration methods
3. **[Development](../development/)** - Learn about extending and customizing integrations
4. **[Troubleshooting](../examples/troubleshooting.md)** - Solve common integration issues

Ready to see it all in action? Check out **[Examples](../examples/)** for complete working implementations.
