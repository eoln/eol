#!/bin/bash
# Sync dependencies across all packages in the workspace

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_info "📦 Syncing dependencies for EOL workspace..."

# Ensure UV is available
if ! command -v uv &> /dev/null; then
    print_warning "UV not found. Running setup script..."
    ./scripts/setup-workspace.sh
    exit 0
fi

# Activate virtual environment if not active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        print_info "Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

# Update UV itself
print_info "Updating UV..."
uv self update || true

# Sync all workspace packages
print_info "Syncing workspace packages..."
uv sync --all-packages

# Apply constraints
print_info "Applying version constraints..."
if [ -f "requirements/constraints.txt" ]; then
    uv pip install -c requirements/constraints.txt -r requirements/base.txt
fi

# Update each package
for package in packages/*; do
    if [ -f "$package/pyproject.toml" ]; then
        package_name=$(basename "$package")
        print_info "Updating $package_name..."
        cd "$package"

        # Reinstall in editable mode
        uv pip install -e . --upgrade

        cd "$PROJECT_ROOT"
    fi
done

# Compile requirements for reproducibility
print_info "Compiling requirements..."
uv pip compile requirements/base.txt -o requirements/base.lock
uv pip compile requirements/dev.txt -o requirements/dev.lock

# Show dependency tree
print_info "Dependency tree:"
uv pip tree || pipdeptree || true

print_success "✅ Dependencies synced successfully!"

# Check for outdated packages
print_info "Checking for outdated packages..."
uv pip list --outdated || true
