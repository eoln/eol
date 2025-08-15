#!/bin/bash

# EOL RAG Context - Automated Test Runner with Redis
# This script automatically starts Redis, runs tests, and stops Redis

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "EOL RAG Context - Automated Test Suite"
echo "======================================"

# Find Python (prefer venv)
if [ -d ".venv" ]; then
    echo "Using virtual environment..."
    source .venv/bin/activate
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
else
    echo "Using system Python..."
    PYTHON_CMD="python3"
fi

# Check for pytest
if ! $PYTHON_CMD -m pytest --version >/dev/null 2>&1; then
    echo -e "${YELLOW}Installing pytest and dependencies...${NC}"
    $PYTHON_CMD -m pip install pytest pytest-asyncio pytest-cov >/dev/null 2>&1 || true
fi

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Function to check if Redis is running
check_redis() {
    redis-cli ping 2>/dev/null | grep -q PONG
}

# Function to start Redis
start_redis() {
    # Check if Redis is already running
    if check_redis; then
        echo -e "${GREEN}Redis is already running${NC}"
        return 0
    fi

    # Try Docker first
    if command -v docker >/dev/null 2>&1; then
        if docker info >/dev/null 2>&1; then
            echo "Starting Redis with Docker..."
            docker rm -f eol-test-redis 2>/dev/null || true
            docker run -d --name eol-test-redis -p 6379:6379 redis/redis-stack:latest >/dev/null 2>&1

            # Wait for Redis to be ready
            for i in {1..30}; do
                if docker exec eol-test-redis redis-cli ping 2>/dev/null | grep -q PONG; then
                    echo -e "${GREEN}Redis is ready (Docker)${NC}"
                    REDIS_MODE="docker"
                    return 0
                fi
                sleep 1
            done
        fi
    fi

    # Try native Redis
    if command -v redis-server >/dev/null 2>&1; then
        echo "Starting Redis natively..."
        redis-server --port 6379 --daemonize yes --save "" --appendonly no >/dev/null 2>&1

        # Wait for Redis to be ready
        for i in {1..30}; do
            if check_redis; then
                echo -e "${GREEN}Redis is ready (Native)${NC}"
                REDIS_MODE="native"
                return 0
            fi
            sleep 1
        done
    fi

    echo -e "${RED}Failed to start Redis${NC}"
    echo "Please install Redis or Docker"
    exit 1
}

# Function to stop Redis
stop_redis() {
    if [ "$REDIS_MODE" = "docker" ]; then
        echo "Stopping Redis container..."
        docker stop eol-test-redis >/dev/null 2>&1 || true
        docker rm eol-test-redis >/dev/null 2>&1 || true
    elif [ "$REDIS_MODE" = "native" ]; then
        echo "Stopping Redis server..."
        redis-cli shutdown >/dev/null 2>&1 || true
    fi
}

# Trap to ensure Redis is stopped on exit
trap stop_redis EXIT

# Start Redis
start_redis

echo ""
echo "======================================"
echo "Running Tests with Coverage"
echo "======================================"
echo ""

# Run all tests (unit + integration)
$PYTHON_CMD -m pytest tests/ \
    --cov=eol.rag_context \
    --cov-report=term \
    --cov-report=html:coverage/html \
    --tb=short \
    -v

EXIT_CODE=$?

# Show coverage summary
echo ""
echo "======================================"
echo "Coverage Summary"
echo "======================================"

$PYTHON_CMD -m pytest tests/ \
    --cov=eol.rag_context \
    --cov-report= \
    --quiet 2>/dev/null | grep -E "TOTAL" || echo "Coverage calculation failed"

# Check if we reached 80%
COVERAGE=$($PYTHON_CMD -c "
import subprocess
import re
result = subprocess.run(
    ['$PYTHON_CMD', '-m', 'pytest', 'tests/', '--cov=eol.rag_context', '--cov-report=', '--quiet'],
    capture_output=True, text=True
)
for line in result.stdout.split('\\n'):
    if 'TOTAL' in line:
        match = re.search(r'(\d+)%', line)
        if match:
            print(match.group(1))
            break
" 2>/dev/null || echo "0")

echo ""
if [ "$COVERAGE" -ge 80 ] 2>/dev/null; then
    echo -e "${GREEN}✅ Coverage target achieved: ${COVERAGE}%${NC}"
else
    echo -e "${YELLOW}⚠️  Coverage below target: ${COVERAGE}% (target: 80%)${NC}"
fi

echo ""
echo "HTML coverage report: coverage/html/index.html"

exit $EXIT_CODE
