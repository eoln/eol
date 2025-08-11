#!/bin/bash

# EOL RAG Context - Integration Test Runner
# This script runs integration tests with proper setup and teardown

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "EOL RAG Context - Integration Test Runner"
echo "========================================"
echo ""

# Function to check Redis
check_redis() {
    if redis-cli ping > /dev/null 2>&1; then
        # Check for RediSearch module
        if redis-cli MODULE LIST | grep -q search; then
            return 0
        else
            echo -e "${YELLOW}Warning: RediSearch module not loaded${NC}"
            return 1
        fi
    else
        return 1
    fi
}

# 1. Check prerequisites
echo "Checking prerequisites..."
echo "-------------------------"

# Check if setup has been run
if [ ! -d ".venv" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Please run ./setup_test_environment.sh first"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Check Redis
if ! check_redis; then
    echo -e "${YELLOW}Redis Stack not running or RediSearch not available${NC}"
    echo "Starting Redis Stack Server..."
    
    # Stop any existing Redis
    redis-cli shutdown 2>/dev/null || true
    sleep 2
    
    # Start Redis Stack
    redis-stack-server --daemonize yes
    sleep 3
    
    if check_redis; then
        echo -e "${GREEN}✓${NC} Redis Stack Server started"
    else
        echo -e "${RED}Failed to start Redis Stack Server${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Redis Stack Server is running"
fi

# Check test data
if [ ! -d "tests/test_data" ]; then
    echo -e "${YELLOW}Test data not found, creating...${NC}"
    mkdir -p tests/test_data
    
    # Create minimal test files
    echo "# Test Document" > tests/test_data/test.md
    echo "def test(): pass" > tests/test_data/test.py
    echo '{"test": true}' > tests/test_data/test.json
    echo "Test content" > tests/test_data/test.txt
fi
echo -e "${GREEN}✓${NC} Test data available"

# 2. Clean Redis data
echo ""
echo "Preparing Redis..."
echo "------------------"

redis-cli FLUSHDB > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Redis data cleared"

# 3. Set Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
echo -e "${GREEN}✓${NC} Python path configured"

# 4. Run integration tests
echo ""
echo "Running Integration Tests..."
echo "=============================="
echo ""

# Create results directory
mkdir -p test_results

# Run tests with detailed output
python -m pytest tests/integration/ \
    --verbose \
    --tb=short \
    --color=yes \
    --capture=no \
    --junit-xml=test_results/integration.xml \
    --cov=eol.rag_context \
    --cov-report=term-missing \
    --cov-report=html:coverage/integration \
    --cov-report=xml:test_results/coverage.xml \
    2>&1 | tee test_results/integration.log

# Capture exit code
TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
echo "Test Results Summary"
echo "========================================"

# Parse test results
if [ -f test_results/integration.xml ]; then
    # Extract test counts from JUnit XML (simple parsing)
    TOTAL=$(grep -o 'tests="[0-9]*"' test_results/integration.xml | head -1 | grep -o '[0-9]*')
    FAILURES=$(grep -o 'failures="[0-9]*"' test_results/integration.xml | head -1 | grep -o '[0-9]*')
    ERRORS=$(grep -o 'errors="[0-9]*"' test_results/integration.xml | head -1 | grep -o '[0-9]*')
    SKIPPED=$(grep -o 'skipped="[0-9]*"' test_results/integration.xml | head -1 | grep -o '[0-9]*' || echo "0")
    
    PASSED=$((TOTAL - FAILURES - ERRORS - SKIPPED))
    
    echo "Total Tests: $TOTAL"
    if [ "$PASSED" -gt 0 ]; then
        echo -e "${GREEN}✓ Passed: $PASSED${NC}"
    fi
    if [ "$SKIPPED" -gt 0 ]; then
        echo -e "${YELLOW}⊘ Skipped: $SKIPPED${NC}"
    fi
    if [ "$FAILURES" -gt 0 ]; then
        echo -e "${RED}✗ Failed: $FAILURES${NC}"
    fi
    if [ "$ERRORS" -gt 0 ]; then
        echo -e "${RED}✗ Errors: $ERRORS${NC}"
    fi
fi

# Coverage summary
echo ""
echo "Coverage Report:"
echo "----------------"
if [ -f test_results/coverage.xml ]; then
    # Extract coverage percentage (simple parsing)
    COV=$(grep -o 'line-rate="[0-9.]*"' test_results/coverage.xml | head -1 | grep -o '[0-9.]*')
    COV_PCT=$(echo "$COV * 100" | bc 2>/dev/null || echo "0")
    COV_INT=${COV_PCT%.*}
    
    if [ "$COV_INT" -ge 80 ]; then
        echo -e "${GREEN}✓ Coverage: ${COV_INT}%${NC}"
    else
        echo -e "${YELLOW}⚠ Coverage: ${COV_INT}% (target: 80%)${NC}"
    fi
    echo "Detailed report: coverage/integration/index.html"
fi

echo ""
echo "Test Artifacts:"
echo "--------------"
echo "• Test results: test_results/integration.xml"
echo "• Test log: test_results/integration.log"
echo "• Coverage HTML: coverage/integration/index.html"
echo "• Coverage XML: test_results/coverage.xml"

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================"
    echo "All integration tests passed!"
    echo "========================================${NC}"
else
    echo -e "${RED}========================================"
    echo "Integration tests failed!"
    echo "========================================${NC}"
    echo ""
    echo "To debug failures:"
    echo "1. Check test_results/integration.log for details"
    echo "2. Run specific test: pytest tests/integration/test_name.py -xvs"
    echo "3. Check Redis: redis-cli ping"
    echo "4. Check modules: redis-cli MODULE LIST"
fi

exit $TEST_EXIT_CODE