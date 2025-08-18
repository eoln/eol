# Claude Desktop MCP Server Setup Guide

## Prerequisites

1. **Redis Stack** must be running:

   ```bash
   # Check if Redis is running
   redis-cli ping
   # Should return: PONG

   # If not running, start it:
   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
   ```

2. **Python environment** activated:

   ```bash
   cd /Users/eoln/Devel/eol/packages/eol-rag-context
   source .venv/bin/activate
   ```

## Setup Instructions

### Step 1: Configure Claude Desktop

1. Open Claude Desktop settings
2. Find the MCP configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

3. Add the EOL RAG Context server configuration:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "/Users/eoln/Devel/eol/packages/eol-rag-context/.venv/bin/python",
      "args": [
        "/Users/eoln/Devel/eol/packages/eol-rag-context/mcp_launcher.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/eoln/Devel/eol/packages/eol-rag-context/src"
      }
    }
  }
}
```

### Step 2: Restart Claude Desktop

After updating the configuration:

1. Completely quit Claude Desktop (Cmd+Q on macOS)
2. Restart Claude Desktop
3. The MCP server should now be available

### Step 3: Verify Connection

In a new Claude conversation, you should be able to use these tools:

- **index_directory** - Index a folder of documents
- **search_context** - Search for relevant information
- **list_sources** - List all indexed sources
- **remove_source** - Remove an indexed source
- **get_stats** - Get indexing statistics

## Testing the Integration

### Test 1: Check Stats

Ask Claude: "Can you check the RAG indexing stats?"

Claude should use the `get_stats` tool and show you:

- Documents indexed
- Chunks created
- Number of sources
- Redis connection status

### Test 2: Index Documents

Ask Claude: "Please index the /Users/eoln/Devel/eol/packages/eol-rag-context/examples directory"

Claude should:

- Use the `index_directory` tool
- Report how many files and chunks were indexed

### Test 3: Search Content

Ask Claude: "Search for information about 'RAG context' in the indexed documents"

Claude should:

- Use the `search_context` tool
- Return relevant results with scores

## Troubleshooting

### MCP Server Not Appearing

1. **Check logs**:

   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

2. **Test server manually**:

   ```bash
   cd /Users/eoln/Devel/eol/packages/eol-rag-context
   source .venv/bin/activate
   python mcp_launcher.py
   ```

   You should see the FastMCP banner

3. **Verify Redis connection**:

   ```bash
   redis-cli ping
   ```

### Common Issues

1. **"Redis connection failed"**
   - Ensure Redis is running: `docker ps | grep redis`
   - Check port 6379 is not blocked

2. **"Module not found"**
   - Activate the virtual environment
   - Install dependencies: `pip install -e .`

3. **"MCP server not responding"**
   - Check Claude Desktop logs
   - Verify Python path in config is correct
   - Ensure mcp_launcher.py is executable

## Available MCP Tools

### 1. index_directory(path, recursive=True)

Indexes all documents in a directory.

**Example**: "Index the folder /path/to/docs recursively"

### 2. search_context(query, max_results=10)

Searches indexed documents for relevant content.

**Example**: "Search for 'machine learning' in the indexed documents"

### 3. list_sources()

Lists all indexed document sources.

**Example**: "Show me all indexed sources"

### 4. remove_source(source_id)

Removes a specific indexed source.

**Example**: "Remove the source with ID abc123"

### 5. get_stats()

Returns current indexing statistics.

**Example**: "What are the current RAG statistics?"

## Advanced Configuration

### Custom Config File

Create `mcp_config.yaml`:

```yaml
redis:
  host: localhost
  port: 6379

embedding:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2

cache:
  enabled: true
  ttl_seconds: 3600
```

Then modify the launcher to use it.

## Next Steps

1. Index your documentation folders
2. Use Claude to search through your indexed content
3. Build a knowledge base by indexing relevant documents
4. Use the semantic cache for faster responses

---

**Note**: The MCP server runs in the background when Claude Desktop starts. It will automatically connect to Redis and be ready to index and search documents.
