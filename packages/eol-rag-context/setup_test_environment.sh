#!/bin/bash

# EOL RAG Context - Test Environment Setup Script
# This script sets up all dependencies needed for running integration tests

set -e  # Exit on error

echo "========================================"
echo "Setting up EOL RAG Context Test Environment"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a brew package is installed
brew_installed() {
    brew list "$1" &>/dev/null
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

echo "1. Checking system dependencies..."
echo "-----------------------------------"

# Check for Homebrew
if command_exists brew; then
    print_status 0 "Homebrew is installed"
else
    print_status 1 "Homebrew is not installed. Please install from https://brew.sh"
    exit 1
fi

# Check for Python 3.11+
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
        print_status 0 "Python $PYTHON_VERSION is installed"
    else
        print_status 1 "Python 3.11+ required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    print_status 1 "Python 3 is not installed"
    exit 1
fi

echo ""
echo "2. Installing system dependencies..."
echo "-----------------------------------"

# Install libmagic for file type detection
if brew_installed libmagic; then
    print_status 0 "libmagic is already installed"
else
    echo "Installing libmagic..."
    brew install libmagic
    print_status $? "libmagic installed"
fi

# Check if Redis Stack is needed
if brew list --cask | grep -q redis-stack-server; then
    print_status 0 "Redis Stack Server is already installed"
else
    echo "Installing Redis Stack Server (includes RediSearch module)..."
    brew tap redis-stack/redis-stack
    brew install --cask redis-stack-server
    print_status $? "Redis Stack Server installed"
fi

echo ""
echo "3. Setting up Python environment..."
echo "-----------------------------------"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    print_status $? "Virtual environment created"
else
    print_status 0 "Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
print_status $? "Virtual environment activated"

echo ""
echo "4. Installing Python dependencies..."
echo "-----------------------------------"

# Upgrade pip
pip install --upgrade pip --quiet

# Install package in development mode
echo "Installing package and dependencies..."
pip install -e . --quiet
print_status $? "Package installed in development mode"

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-asyncio pytest-cov --quiet
print_status $? "Test dependencies installed"

# Install additional dependencies for integration tests
echo "Installing integration test dependencies..."
pip install \
    python-magic \
    beautifulsoup4 \
    markdown \
    pyyaml \
    lxml \
    html5lib \
    --quiet
print_status $? "Integration test dependencies installed"

echo ""
echo "5. Starting Redis Stack Server..."
echo "-----------------------------------"

# Stop any existing Redis instances
if pgrep -x "redis-server" > /dev/null; then
    echo "Stopping existing Redis server..."
    redis-cli shutdown 2>/dev/null || true
    sleep 2
fi

if pgrep -x "redis-stack-server" > /dev/null; then
    echo "Redis Stack Server is already running"
else
    echo "Starting Redis Stack Server..."
    redis-stack-server --daemonize yes
    sleep 3
fi

# Verify Redis is running
if redis-cli ping > /dev/null 2>&1; then
    print_status 0 "Redis Stack Server is running"

    # Check for RediSearch module
    if redis-cli MODULE LIST | grep -q search; then
        print_status 0 "RediSearch module is loaded"
    else
        print_status 1 "RediSearch module is not loaded"
        exit 1
    fi
else
    print_status 1 "Failed to start Redis Stack Server"
    exit 1
fi

echo ""
echo "6. Creating test data..."
echo "-----------------------------------"

# Create test data directory
TEST_DATA_DIR="tests/test_data"
if [ ! -d "$TEST_DATA_DIR" ]; then
    mkdir -p "$TEST_DATA_DIR"
    print_status $? "Test data directory created"
else
    print_status 0 "Test data directory exists"
fi

# Create sample test files
cat > "$TEST_DATA_DIR/sample.md" << 'EOF'
# Sample Markdown Document

This is a test document for the RAG system.

## Features

- Hierarchical indexing
- Vector search
- Semantic caching

## Code Example

```python
def hello_world():
    return "Hello, World!"
```

## Conclusion

This document is used for integration testing.
EOF

cat > "$TEST_DATA_DIR/sample.py" << 'EOF'
"""Sample Python module for testing."""

def calculate_factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        self.result = a + b
        return self.result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        self.result = a * b
        return self.result
EOF

cat > "$TEST_DATA_DIR/config.json" << 'EOF'
{
    "name": "test-project",
    "version": "1.0.0",
    "settings": {
        "debug": true,
        "timeout": 30,
        "features": {
            "rag": true,
            "cache": true,
            "graph": true
        }
    },
    "dependencies": {
        "redis": "^8.0.0",
        "fastmcp": "^0.1.0"
    }
}
EOF

cat > "$TEST_DATA_DIR/sample.txt" << 'EOF'
This is a plain text file for testing document processing.

It contains multiple paragraphs with various content that should be
properly chunked and indexed by the RAG system.

The system should handle:
- Text extraction
- Proper chunking
- Metadata extraction
- Hierarchical organization

Each paragraph should be processed according to the configured
chunking strategy, preserving semantic meaning while optimizing
for the LLM context window.
EOF

print_status $? "Test data files created"

echo ""
echo "7. Running verification tests..."
echo "-----------------------------------"

# Test Redis connection
python3 << 'EOF'
import sys
try:
    import redis
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print("✓ Redis connection successful")
except Exception as e:
    print(f"✗ Redis connection failed: {e}")
    sys.exit(1)
EOF

# Test file type detection
python3 << 'EOF'
import sys
try:
    import magic
    m = magic.Magic(mime=True)
    result = m.from_file("tests/test_data/sample.md")
    if "text" in result:
        print("✓ File type detection working")
    else:
        print(f"✗ File type detection returned: {result}")
        sys.exit(1)
except Exception as e:
    print(f"✗ File type detection failed: {e}")
    sys.exit(1)
EOF

# Test BeautifulSoup
python3 << 'EOF'
import sys
try:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup("<h1>Test</h1>", "html.parser")
    if soup.h1.text == "Test":
        print("✓ BeautifulSoup working")
    else:
        print("✗ BeautifulSoup parsing failed")
        sys.exit(1)
except Exception as e:
    print(f"✗ BeautifulSoup failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "========================================"
echo -e "${GREEN}Test environment setup complete!${NC}"
echo "========================================"
echo ""
echo "You can now run tests with:"
echo "  ./run_integration_tests.sh     # Run integration tests only"
echo "  ./test_all.sh                   # Run all tests with coverage"
echo ""
echo "Redis Stack Server is running on port 6379"
echo "To stop Redis: redis-cli shutdown"
echo ""
