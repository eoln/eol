#!/bin/bash
# Local setup script for eol-rag-context MCP server (no sudo required)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}EOL RAG Context MCP Server Setup (Local)${NC}"
echo -e "${GREEN}==================================================${NC}"

PROJECT_DIR="/Users/eoln/Devel/eol/packages/eol-rag-context"
VENV_DIR="/Users/eoln/Devel/eol/.venv"

# Create local bin directory
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# Check if local bin is in PATH
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo -e "${YELLOW}Note: Add $LOCAL_BIN to your PATH:${NC}"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Create wrapper script in user's local bin
WRAPPER_SCRIPT="$LOCAL_BIN/eol-rag-mcp"
cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
# EOL RAG Context MCP Server Wrapper

export PYTHONPATH="$PROJECT_DIR/src"
exec "$VENV_DIR/bin/python" "$PROJECT_DIR/mcp_launcher_final.py" "\$@"
EOF

chmod +x "$WRAPPER_SCRIPT"
echo -e "${GREEN}Created wrapper script at: $WRAPPER_SCRIPT${NC}"

# Create MCP config for Claude Code
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
        "PYTHONPATH": "$PROJECT_DIR/src"
      }
    }
  }
}
EOF

echo -e "${GREEN}Created MCP config at: $MCP_CONFIG_DIR/mcp-settings.json${NC}"

# Create alias for easy access
SHELL_RC="$HOME/.zshrc"
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "alias eol-rag=" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# EOL RAG Context MCP Server" >> "$SHELL_RC"
    echo "alias eol-rag='$WRAPPER_SCRIPT'" >> "$SHELL_RC"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
    echo -e "${GREEN}Added alias to $SHELL_RC${NC}"
fi

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}==================================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Reload your shell: source $SHELL_RC"
echo "2. Test the MCP server: eol-rag"
echo "3. Use with Claude Code CLI"
echo ""
echo -e "${YELLOW}Claude Code Usage Examples:${NC}"
echo "  claude 'Index this project for semantic search'"
echo "  claude 'Find all database connection code'"
echo "  claude 'Show indexing statistics'"
