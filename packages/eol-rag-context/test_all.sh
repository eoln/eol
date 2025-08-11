#!/bin/bash
# Complete test runner for EOL RAG Context
# Runs all tests with proper Redis Stack setup

set -e

echo "======================================"
echo "EOL RAG Context - Complete Test Suite"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Redis with RediSearch
check_redis_stack() {
    if redis-cli ping >/dev/null 2>&1; then
        if redis-cli MODULE LIST | grep -q search; then
            return 0
        fi
    fi
    return 1
}

# Function to stop Redis
stop_redis() {
    echo "Cleaning up Redis..."
    redis-cli shutdown 2>/dev/null || true
}

# Trap to ensure cleanup on exit
trap stop_redis EXIT INT TERM

# Check Python
if ! command_exists python3 && ! command_exists python; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

PYTHON_CMD=$(command_exists python3 && echo "python3" || echo "python")

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Please run ./setup_test_environment.sh first"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
MISSING_DEPS=""

# Check for critical Python packages
for pkg in redis pytest pytest_asyncio pytest_cov numpy pydantic; do
    if ! $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done

if [ ! -z "$MISSING_DEPS" ]; then
    echo -e "${YELLOW}Installing missing packages:$MISSING_DEPS${NC}"
    pip install $MISSING_DEPS
fi

# Start Redis Stack if needed
echo ""
echo "Checking Redis Stack..."

if ! check_redis_stack; then
    echo "Starting Redis Stack Server..."
    
    # Stop any existing Redis
    redis-cli shutdown 2>/dev/null || true
    sleep 2
    
    # Start Redis Stack (has RediSearch module)
    if command_exists redis-stack-server; then
        redis-stack-server --daemonize yes
        sleep 3
        
        if check_redis_stack; then
            echo -e "${GREEN}✓ Redis Stack Server started${NC}"
        else
            echo -e "${RED}Failed to start Redis Stack Server${NC}"
            echo "Please install Redis Stack: brew install --cask redis-stack-server"
            exit 1
        fi
    else
        echo -e "${RED}Redis Stack Server not installed${NC}"
        echo "Please install with: brew install --cask redis-stack-server"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Redis Stack is already running${NC}"
fi

# Clear Redis data for clean test state
redis-cli FLUSHDB >/dev/null 2>&1
echo -e "${GREEN}✓ Redis data cleared${NC}"

# Set environment variables
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Create test data if needed
if [ ! -d "tests/test_data" ]; then
    mkdir -p tests/test_data
    echo "# Test Document" > tests/test_data/test.md
    echo "def test(): pass" > tests/test_data/test.py
    echo '{"test": true}' > tests/test_data/test.json
    echo "Test content" > tests/test_data/test.txt
fi

# Run tests
echo ""
echo "======================================"
echo "Running Test Suite"
echo "======================================"
echo ""

# Create results directory
mkdir -p test_results
mkdir -p coverage

# Run unit tests first
echo "Running Unit Tests..."
$PYTHON_CMD -m pytest \
    tests/test_config.py \
    tests/test_embeddings.py \
    tests/test_document_processor.py \
    tests/test_indexer.py \
    tests/test_mcp_server.py \
    --cov=eol.rag_context \
    --cov-branch \
    --cov-report= \
    --tb=short \
    -q 2>/dev/null

UNIT_EXIT=$?

if [ $UNIT_EXIT -eq 0 ]; then
    UNIT_STATUS="${GREEN}PASSED${NC}"
else
    UNIT_STATUS="${RED}FAILED${NC}"
fi

# Run integration tests
echo ""
echo "Running Integration Tests..."
$PYTHON_CMD -m pytest \
    tests/integration/ \
    --cov=eol.rag_context \
    --cov-branch \
    --cov-append \
    --cov-report= \
    --tb=short \
    -q 2>/dev/null

INTEGRATION_EXIT=$?

if [ $INTEGRATION_EXIT -eq 0 ]; then
    INTEGRATION_STATUS="${GREEN}PASSED${NC}"
else
    INTEGRATION_STATUS="${YELLOW}SOME FAILED${NC}"
fi

# Generate combined coverage report
echo ""
echo "======================================"
echo "Coverage Report"
echo "======================================"
echo ""

# Generate coverage reports in multiple formats
$PYTHON_CMD -m pytest \
    --cov=eol.rag_context \
    --cov-report=term:skip-covered \
    --cov-report=html:coverage/html \
    --cov-report=xml:test_results/coverage.xml \
    --cov-report=json:test_results/coverage.json \
    --quiet \
    --no-header \
    tests/ 2>/dev/null

# Extract coverage percentage from JSON
COVERAGE=$($PYTHON_CMD -c "
import json
import sys
try:
    with open('test_results/coverage.json', 'r') as f:
        data = json.load(f)
        coverage = data.get('totals', {}).get('percent_covered', 0)
        print(f'{coverage:.1f}')
except:
    print('0')
" 2>/dev/null || echo "0")

echo ""
echo "======================================"
echo "Test Results Summary"
echo "======================================"
echo ""

# Parse coverage as integer for comparison
COVERAGE_INT=$(echo "$COVERAGE" | cut -d'.' -f1)

if [ "$COVERAGE_INT" -ge 80 ]; then
    echo -e "${GREEN}✅ Coverage: ${COVERAGE}% - Target (80%) achieved!${NC}"
    COVERAGE_MET=true
else
    echo -e "${YELLOW}⚠️  Coverage: ${COVERAGE}% - Below target (80%)${NC}"
    COVERAGE_MET=false
fi

echo -e "Unit Tests: $UNIT_STATUS"
echo -e "Integration Tests: $INTEGRATION_STATUS"
echo ""
echo "Detailed Reports:"
echo "• HTML Coverage: coverage/html/index.html"
echo "• XML Coverage: test_results/coverage.xml"
echo "• JSON Coverage: test_results/coverage.json"
echo ""

# Determine overall exit code
if [ $UNIT_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Test suite failed: Unit tests failed${NC}"
    exit $UNIT_EXIT
elif [ "$COVERAGE_MET" = false ]; then
    echo -e "${YELLOW}⚠️  Tests passed but coverage is below 80% target${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All tests passed with adequate coverage!${NC}"
    exit 0
fi