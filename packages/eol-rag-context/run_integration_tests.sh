#!/bin/bash

# Run integration tests for EOL RAG Context
set -e

echo "======================================"
echo "EOL RAG Context - Integration Tests"
echo "======================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Parse arguments
COVERAGE_REPORT=${1:-term}
PYTEST_ARGS=${2:-}

echo "Starting test environment..."
echo ""

# Clean up any existing containers
docker-compose -f docker-compose.test.yml down --volumes --remove-orphans 2>/dev/null || true

# Start Redis
echo "Starting Redis..."
docker-compose -f docker-compose.test.yml up -d redis

# Wait for Redis to be healthy
echo "Waiting for Redis to be ready..."
for i in {1..30}; do
    if docker-compose -f docker-compose.test.yml exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
        echo "Redis is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: Redis failed to start"
        docker-compose -f docker-compose.test.yml logs redis
        exit 1
    fi
    sleep 1
done

echo ""
echo "Running integration tests..."
echo ""

# Run tests locally (not in Docker for now, due to build complexity)
export REDIS_HOST=localhost
export REDIS_PORT=6379
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import redis" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install redis aioredis numpy pydantic pydantic-settings \
                sentence-transformers aiofiles beautifulsoup4 markdown \
                pyyaml networkx watchdog gitignore-parser \
                pytest pytest-asyncio pytest-cov pytest-timeout
fi

# Run integration tests
python -m pytest tests/integration/ \
    -xvs \
    --cov=eol.rag_context \
    --cov-report=$COVERAGE_REPORT \
    --cov-report=html:coverage/html \
    --tb=short \
    -m integration \
    $PYTEST_ARGS

TEST_EXIT_CODE=$?

echo ""
echo "======================================"
echo "Test Results"
echo "======================================"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All integration tests passed!"
    
    # Show coverage summary
    echo ""
    echo "Coverage Report:"
    python -m pytest tests/integration/ tests/ \
        --cov=eol.rag_context \
        --cov-report=term:skip-covered \
        --quiet \
        --no-header 2>/dev/null | grep -E "TOTAL|Name.*Cover" || true
        
    echo ""
    echo "HTML coverage report generated at: coverage/html/index.html"
else
    echo "❌ Some tests failed. Exit code: $TEST_EXIT_CODE"
fi

echo ""
echo "Cleaning up..."

# Stop services
docker-compose -f docker-compose.test.yml down --volumes

echo "Done!"
exit $TEST_EXIT_CODE