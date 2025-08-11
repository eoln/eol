#!/bin/bash
# Run tests for all packages in the workspace

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🧪 Running tests for all EOL packages..."
echo "========================================="

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        print_error "Virtual environment not found. Run ./scripts/setup-workspace.sh first."
        exit 1
    fi
fi

# Track test results
FAILED_PACKAGES=""
TOTAL_PACKAGES=0
PASSED_PACKAGES=0

# Run tests for each package
for package in packages/*; do
    if [ -f "$package/pyproject.toml" ] && [ -d "$package/tests" ]; then
        package_name=$(basename "$package")
        ((TOTAL_PACKAGES++))
        
        echo ""
        print_info "Testing $package_name..."
        echo "----------------------------------------"
        
        cd "$package"
        
        # Run tests with coverage
        if pytest tests/ \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=html:coverage/html \
            -v \
            --tb=short \
            2>&1; then
            print_success "$package_name tests passed!"
            ((PASSED_PACKAGES++))
        else
            print_error "$package_name tests failed!"
            FAILED_PACKAGES="$FAILED_PACKAGES $package_name"
        fi
        
        cd "$PROJECT_ROOT"
    else
        print_warning "Skipping $(basename "$package") - no tests found"
    fi
done

# Run integration tests if available
if [ -d "tests/integration" ]; then
    echo ""
    print_info "Running integration tests..."
    echo "----------------------------------------"
    
    if pytest tests/integration/ -v --tb=short 2>&1; then
        print_success "Integration tests passed!"
    else
        print_error "Integration tests failed!"
        FAILED_PACKAGES="$FAILED_PACKAGES integration"
    fi
fi

# Generate combined coverage report
echo ""
print_info "Generating combined coverage report..."
if command -v coverage &> /dev/null; then
    coverage combine packages/*/coverage/.coverage* 2>/dev/null || true
    coverage report || true
    coverage html -d coverage/html || true
    print_info "Coverage report available at: coverage/html/index.html"
fi

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
print_info "Total packages tested: $TOTAL_PACKAGES"
print_success "Passed: $PASSED_PACKAGES"

if [ -n "$FAILED_PACKAGES" ]; then
    print_error "Failed packages:$FAILED_PACKAGES"
    echo ""
    print_error "❌ Some tests failed. Please review the output above."
    exit 1
else
    echo ""
    print_success "✅ All tests passed successfully!"
fi

# Run linting and type checking
echo ""
print_info "Running code quality checks..."
echo "----------------------------------------"

# Black formatting check
if command -v black &> /dev/null; then
    print_info "Checking code formatting with Black..."
    black --check packages/*/src || print_warning "Some files need formatting. Run: black packages/*/src"
fi

# Ruff linting
if command -v ruff &> /dev/null; then
    print_info "Running Ruff linter..."
    ruff check packages/*/src || print_warning "Linting issues found. Run: ruff check --fix packages/*/src"
fi

# MyPy type checking
if command -v mypy &> /dev/null; then
    print_info "Running MyPy type checker..."
    mypy packages/*/src || print_warning "Type checking issues found."
fi

print_success "✅ Test suite complete!"