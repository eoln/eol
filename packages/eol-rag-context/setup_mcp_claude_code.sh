#!/bin/bash
# Setup script for eol-rag-context MCP server with Claude Code CLI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}EOL RAG Context MCP Server Setup for Claude Code${NC}"
echo -e "${GREEN}==================================================${NC}"

# Check if running from correct directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the eol-rag-context directory${NC}"
    exit 1
fi

PROJECT_DIR=$(pwd)
VENV_DIR="/Users/eoln/Devel/eol/.venv"

# Step 1: Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Step 2: Check if uv is installed
echo -e "\n${YELLOW}Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed. Please install it first:${NC}"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv is installed"

# Step 3: Install dependencies
echo -e "\n${YELLOW}Installing dependencies with uv...${NC}"
cd /Users/eoln/Devel/eol
uv sync

# Step 4: Check Redis
echo -e "\n${YELLOW}Checking Redis...${NC}"
if ! redis-cli ping &> /dev/null; then
    echo -e "${RED}Redis is not running!${NC}"
    echo "Starting Redis with custom config..."
    /usr/local/opt/redis/bin/redis-server "$PROJECT_DIR/redis-v8.conf" &
    sleep 2
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}Redis started successfully${NC}"
    else
        echo -e "${RED}Failed to start Redis${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Redis is running${NC}"
    REDIS_VERSION=$(redis-cli INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')
    echo "Redis version: $REDIS_VERSION"
fi

# Step 5: Create MCP config for Claude Code
echo -e "\n${YELLOW}Creating MCP configuration...${NC}"

MCP_CONFIG_DIR="$HOME/.config/claude-code"
mkdir -p "$MCP_CONFIG_DIR"

cat > "$MCP_CONFIG_DIR/mcp-settings.json" << EOF
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "$VENV_DIR/bin/python",
      "args": [
        "$PROJECT_DIR/mcp_launcher_final.py"
      ],
      "env": {
        "PYTHONPATH": "$PROJECT_DIR/src",
        "EOL_RAG_DATA_DIR": "\${PROJECT_ROOT}/.rag-data",
        "EOL_RAG_INDEX_DIR": "\${PROJECT_ROOT}/.rag-index"
      }
    }
  }
}
EOF

echo -e "${GREEN}Created MCP config at: $MCP_CONFIG_DIR/mcp-settings.json${NC}"

# Step 6: Create a wrapper script for global usage
echo -e "\n${YELLOW}Creating global wrapper script...${NC}"

WRAPPER_SCRIPT="/usr/local/bin/eol-rag-mcp"
sudo tee "$WRAPPER_SCRIPT" > /dev/null << EOF
#!/bin/bash
# EOL RAG Context MCP Server Wrapper

export PYTHONPATH="$PROJECT_DIR/src"
exec "$VENV_DIR/bin/python" "$PROJECT_DIR/mcp_launcher_final.py" "\$@"
EOF

sudo chmod +x "$WRAPPER_SCRIPT"
echo -e "${GREEN}Created wrapper script at: $WRAPPER_SCRIPT${NC}"

# Step 7: Test the MCP server
echo -e "\n${YELLOW}Testing MCP server...${NC}"
if timeout 5 "$VENV_DIR/bin/python" "$PROJECT_DIR/mcp_launcher_final.py" 2>&1 | grep -q "Starting MCP RAG Context Server"; then
    echo -e "${GREEN}MCP server test successful!${NC}"
else
    echo -e "${YELLOW}Warning: Could not verify MCP server startup${NC}"
fi

# Step 8: Create project-specific launcher
echo -e "\n${YELLOW}Creating project launcher template...${NC}"

cat > "$PROJECT_DIR/project_mcp_launcher.sh" << 'EOF'
#!/bin/bash
# Project-specific MCP launcher
# Copy this to your project root and customize as needed

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create project-specific directories if they don't exist
mkdir -p "$PROJECT_ROOT/.rag-data"
mkdir -p "$PROJECT_ROOT/.rag-index"

# Export environment variables
export EOL_RAG_DATA_DIR="$PROJECT_ROOT/.rag-data"
export EOL_RAG_INDEX_DIR="$PROJECT_ROOT/.rag-index"

# Launch the MCP server
exec eol-rag-mcp "$@"
EOF

chmod +x "$PROJECT_DIR/project_mcp_launcher.sh"
echo -e "${GREEN}Created project launcher template${NC}"

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}==================================================${NC}"

echo -e "\n${YELLOW}Usage Instructions:${NC}"
echo "1. For Claude Code CLI, the MCP server is now configured globally"
echo "2. For project-specific usage, copy project_mcp_launcher.sh to your project"
echo "3. The MCP server will create .rag-data and .rag-index in each project"
echo ""
echo -e "${YELLOW}Available MCP Tools:${NC}"
echo "  - get_stats: Get indexing statistics"
echo "  - list_sources: List indexed sources"
echo "  - index_directory: Index a directory (with force_reindex option)"
echo "  - search_context: Search indexed content"
echo "  - test_sandbox: Test environment access"
echo ""
echo -e "${YELLOW}Example commands in Claude Code:${NC}"
echo "  claude 'Index the current directory for RAG search'"
echo "  claude 'Search for authentication implementation'"
echo "  claude 'Show me all indexed files'"
