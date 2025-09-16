#!/bin/bash
# Setup UV workspace for EOL monorepo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_info "🚀 Setting up EOL workspace with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    print_warning "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        print_error "Failed to install UV. Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

print_info "UV version: $(uv --version)"

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_info "Project root: $PROJECT_ROOT"

# Clean existing virtual environments
if [ -d ".venv" ]; then
    print_warning "Removing existing .venv..."
    rm -rf .venv
fi

# Initialize UV workspace
print_info "Initializing UV workspace..."
uv venv --python 3.11

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate

# Sync workspace dependencies
print_info "Syncing workspace dependencies..."
uv sync --all-packages

# Install development dependencies
print_info "Installing development dependencies..."
uv pip install -r requirements/dev.txt

# Install package in editable mode
print_info "Installing packages in editable mode..."
for package in packages/*; do
    if [ -f "$package/pyproject.toml" ]; then
        package_name=$(basename "$package")
        print_info "Installing $package_name..."
        uv pip install -e "$package"
    fi
done

# Verify installation
print_info "Verifying installation..."
python -c "import sys; print(f'Python: {sys.version}')"
uv pip list | grep -E "(eol-|redis|pydantic|fastmcp)" || true

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << EOF
# EOL Workspace Environment Variables
PYTHONPATH=\${PYTHONPATH}:packages/eol-core/src:packages/eol-cli/src:packages/eol-rag-context/src
UV_WORKSPACE=true
UV_CACHE_DIR=.uv-cache
EOF
fi

print_success "✅ EOL workspace setup complete!"
print_info ""
print_info "Next steps:"
print_info "  1. Activate the virtual environment: source .venv/bin/activate"
print_info "  2. Run tests: ./scripts/test-all.sh"
print_info "  3. Check dependencies: ./scripts/check-deps.sh"
print_info ""
print_info "For package-specific work:"
print_info "  cd packages/eol-rag-context"
print_info "  uv pip install -e ."
print_info ""
