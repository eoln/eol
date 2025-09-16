#!/bin/bash

# EOL RAG Context Test Runner

set -e

echo "======================================"
echo "EOL RAG Context MCP Server Test Suite"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Redis is required
REDIS_TESTS=false
INTEGRATION_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --redis)
            REDIS_TESTS=true
            shift
            ;;
        --integration)
            INTEGRATION_TESTS=true
            REDIS_TESTS=true
            shift
            ;;
        --all)
            REDIS_TESTS=true
            INTEGRATION_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--redis] [--integration] [--all]"
            exit 1
            ;;
    esac
done

# Function to check if Redis is running
check_redis() {
    if redis-cli ping > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start Redis with Docker
start_redis_docker() {
    echo -e "${YELLOW}Starting Redis with Docker...${NC}"
    docker-compose -f docker-compose.test.yml up -d

    # Wait for Redis to be ready
    echo "Waiting for Redis to be ready..."
    for i in {1..30}; do
        if check_redis; then
            echo -e "${GREEN}Redis is ready!${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}Redis failed to start${NC}"
    return 1
}

# Function to stop Redis Docker
stop_redis_docker() {
    echo -e "${YELLOW}Stopping Redis Docker...${NC}"
    docker-compose -f docker-compose.test.yml down
}

# Run unit tests
run_unit_tests() {
    echo ""
    echo "Running Unit Tests..."
    echo "===================="
    pytest tests/test_document_processor.py tests/test_indexer.py tests/test_mcp_server.py -v
}

# Run integration tests
run_integration_tests() {
    echo ""
    echo "Running Integration Tests..."
    echo "==========================="
    pytest tests/test_integration.py --redis -v
}

# Main test execution
echo "Test Configuration:"
echo "  Redis Tests: $REDIS_TESTS"
echo "  Integration Tests: $INTEGRATION_TESTS"
echo ""

# Install test dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -e ".[dev]" -q

# Start Redis if needed
REDIS_STARTED=false
if [ "$REDIS_TESTS" = true ]; then
    if ! check_redis; then
        echo -e "${YELLOW}Redis is not running.${NC}"
        read -p "Start Redis with Docker? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if start_redis_docker; then
                REDIS_STARTED=true
            else
                echo -e "${RED}Failed to start Redis. Skipping Redis tests.${NC}"
                REDIS_TESTS=false
            fi
        else
            echo -e "${YELLOW}Skipping Redis tests.${NC}"
            REDIS_TESTS=false
        fi
    else
        echo -e "${GREEN}Redis is already running.${NC}"
    fi
fi

# Run tests
echo ""
echo "Starting test execution..."
echo ""

# Always run unit tests
run_unit_tests
UNIT_RESULT=$?

# Run integration tests if requested and Redis is available
INTEGRATION_RESULT=0
if [ "$INTEGRATION_TESTS" = true ] && [ "$REDIS_TESTS" = true ]; then
    run_integration_tests
    INTEGRATION_RESULT=$?
fi

# Stop Redis if we started it
if [ "$REDIS_STARTED" = true ]; then
    stop_redis_docker
fi

# Summary
echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"

if [ $UNIT_RESULT -eq 0 ]; then
    echo -e "Unit Tests: ${GREEN}PASSED${NC}"
else
    echo -e "Unit Tests: ${RED}FAILED${NC}"
fi

if [ "$INTEGRATION_TESTS" = true ]; then
    if [ $INTEGRATION_RESULT -eq 0 ]; then
        echo -e "Integration Tests: ${GREEN}PASSED${NC}"
    else
        echo -e "Integration Tests: ${RED}FAILED${NC}"
    fi
fi

# Exit with appropriate code
if [ $UNIT_RESULT -ne 0 ] || [ $INTEGRATION_RESULT -ne 0 ]; then
    exit 1
fi

exit 0
